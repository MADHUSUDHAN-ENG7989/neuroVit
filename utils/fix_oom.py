import json

def lines(code):
    parts = code.split('\n')
    result = []
    for j, p in enumerate(parts):
        if j < len(parts) - 1:
            result.append(p + '\n')
        else:
            if p:
                result.append(p)
    return result

def make_cell(source_code, cell_type='code'):
    if cell_type == 'code':
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_code if isinstance(source_code, list) else [source_code]
        }

def patch():
    with open('pdscnn_base_paper.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # ---------------------------------------------------------------
    # Fix 1: BATCH_SIZE 64 -> 32 in cell 11 to avoid OOM
    # ---------------------------------------------------------------
    src11 = "".join(cells[11].get('source', []))
    src11 = src11.replace(
        "BATCH_SIZE = 64   # increased from 16 -> 64 for faster throughput on T4",
        "BATCH_SIZE = 32   # T4 has 15 GB VRAM; 32 is safe with PDSCNN + MixUp double pass avoided"
    )
    cells[11]['source'] = lines(src11)

    # ---------------------------------------------------------------
    # Fix 2: Remove the extra clean forward pass in the training loop
    #         Use MixUp-aware accuracy estimate instead (no 2nd pass)
    # ---------------------------------------------------------------
    training_loop_code = """\
DRIVE_SAVE_DIR = '/content/drive/MyDrive/pdscnn_standalone_v2'
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

BEST_CKPT_PATH  = os.path.join(DRIVE_SAVE_DIR, 'pdscnn_best.pth')
FINAL_CKPT_PATH = os.path.join(DRIVE_SAVE_DIR, 'pdscnn_final.pth')
SWA_CKPT_PATH   = os.path.join(DRIVE_SAVE_DIR, 'pdscnn_swa.pth')
HISTORY_PATH    = os.path.join(DRIVE_SAVE_DIR, 'training_history.csv')
CONFIG_PATH     = os.path.join(DRIVE_SAVE_DIR, 'config.json')

EPOCHS          = 100
LR              = 1e-3    # OneCycleLR handles the warm-up so start here
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.10
MIXUP_ALPHA     = 0.4
SWA_START       = 75     # start SWA after epoch 75

optimizer = torch.optim.AdamW(pdscnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=SWA_START,
    pct_start=0.10,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1e4,
)
criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=LABEL_SMOOTHING)
scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

# SWA for final accuracy boost
swa_model     = swa_utils.AveragedModel(pdscnn)
swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=5e-5, anneal_epochs=5)

history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
best_val_acc = 0.0

for epoch in range(EPOCHS):
    pdscnn.train()
    running_loss, running_correct, n_seen = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
    use_scheduler_step = epoch < SWA_START

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # MixUp augmentation
        mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=MIXUP_ALPHA)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits, _ = pdscnn(mixed_imgs)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(pdscnn.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        if use_scheduler_step:
            scheduler.step()

        running_loss += loss.item() * imgs.size(0)
        # Approximate train acc via MixUp-weighted prediction (avoids a 2nd forward pass)
        with torch.no_grad():
            pred = logits.detach().argmax(1)
            correct = lam * (pred == y_a).float().sum() + (1 - lam) * (pred == y_b).float().sum()
            running_correct += correct.item()
        n_seen += imgs.size(0)
        pbar.set_postfix(loss=running_loss / n_seen, acc=running_correct / n_seen)

    train_loss = running_loss / n_seen
    train_acc  = running_correct / n_seen

    # SWA update
    if epoch >= SWA_START:
        swa_model.update_parameters(pdscnn)
        swa_scheduler.step()

    # Validation (no MixUp, no gradient)
    pdscnn.eval()
    val_loss, val_correct, n_val = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits, _ = pdscnn(imgs)
                loss = criterion(logits, labels)
            val_loss    += loss.item() * imgs.size(0)
            val_correct += (logits.argmax(1) == labels).sum().item()
            n_val       += imgs.size(0)

    val_loss /= n_val
    val_acc   = val_correct / n_val
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1}/{EPOCHS} | train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
          f"| val_loss {val_loss:.4f} val_acc {val_acc:.4f} | lr {current_lr:.2e}")

    history['epoch'].append(epoch + 1)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': pdscnn.state_dict(),
            'epoch': epoch + 1,
            'val_acc': val_acc,
            'class_names': CLASS_NAMES,
            'dataset_mean': DATASET_MEAN,
            'dataset_std': DATASET_STD,
            'img_size': IMG_SIZE,
        }, BEST_CKPT_PATH)
        print(f"  -> New best val_acc {val_acc:.4f}, checkpoint saved!")

    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

# Finalise SWA: update BatchNorm stats then save
print("Updating SWA BatchNorm stats...")
torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
torch.save(swa_model.state_dict(), SWA_CKPT_PATH)
print(f"SWA model saved to {SWA_CKPT_PATH}")

torch.save({
    'model_state_dict': pdscnn.state_dict(),
    'epoch': EPOCHS,
    'val_acc': val_acc,
    'class_names': CLASS_NAMES,
    'dataset_mean': DATASET_MEAN,
    'dataset_std': DATASET_STD,
    'img_size': IMG_SIZE,
}, FINAL_CKPT_PATH)

with open(CONFIG_PATH, 'w') as cfg_f:
    json.dump({
        'class_names': CLASS_NAMES,
        'dataset_mean': DATASET_MEAN,
        'dataset_std': DATASET_STD,
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'lr': LR,
        'weight_decay': WEIGHT_DECAY,
        'label_smoothing': LABEL_SMOOTHING,
        'mixup_alpha': MIXUP_ALPHA,
        'swa_start': SWA_START,
        'best_val_acc': best_val_acc,
        'num_train': len(train_df), 'num_val': len(val_df), 'num_test': len(test_df),
    }, cfg_f, indent=2)

print(f"\\nTraining complete. Best val_acc: {best_val_acc:.4f}")
print(f"All outputs saved to: {DRIVE_SAVE_DIR}")
"""
    cells[17] = make_cell(lines(training_loop_code))

    nb['cells'] = cells

    with open('pdscnn_base_paper.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

if __name__ == '__main__':
    patch()
    print("OOM fix applied successfully.")
