import json
import copy

def make_cell(source_code, cell_type='code'):
    if cell_type == 'code':
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_code if isinstance(source_code, list) else [source_code]
        }
    else:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source_code if isinstance(source_code, list) else [source_code]
        }

def lines(code):
    """Split multi-line code string into list of lines (each with trailing newline except last)."""
    parts = code.split('\n')
    result = []
    for j, p in enumerate(parts):
        if j < len(parts) - 1:
            result.append(p + '\n')
        else:
            if p:  # avoid trailing empty string
                result.append(p)
    return result

def patch_notebook():
    with open('pdscnn_base_paper.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # ---------------------------------------------------------------
    # CELL 5: Replace data loading with Drive->local copy + fast load
    # ---------------------------------------------------------------
    load_code = """\
from google.colab import drive
import os, glob, shutil
import pandas as pd

drive.mount('/content/drive')

DRIVE_PROCESSED_ROOT = '/content/drive/MyDrive/processed_data_vit'
LOCAL_PROCESSED_ROOT = '/content/processed_data_vit'

if not os.path.exists(LOCAL_PROCESSED_ROOT):
    print("Copying preprocessed images from Drive to local SSD (faster I/O during training)...")
    shutil.copytree(DRIVE_PROCESSED_ROOT, LOCAL_PROCESSED_ROOT)
    print("Copy complete!")
else:
    print("Local cache already exists, skipping copy.")

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

records = []
for cls in CLASS_NAMES:
    cls_path = os.path.join(LOCAL_PROCESSED_ROOT, cls)
    if os.path.isdir(cls_path):
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                records.append({'processed_path': os.path.join(cls_path, fname), 'label': cls})

df = pd.DataFrame(records)
print(f"Loaded {len(df)} preprocessed images from local disk.")
print(df['label'].value_counts())

if len(df) == 0:
    raise ValueError('No images found! Make sure they are saved to ' + DRIVE_PROCESSED_ROOT)
"""
    cells[5] = make_cell(lines(load_code))

    # ---------------------------------------------------------------
    # CELL 7: Hardcode known mean/std to save 60s of I/O on startup
    # ---------------------------------------------------------------
    norm_code = """\
# Dataset-specific normalization stats (pre-computed from the processed dataset).
# Hardcoded to avoid re-scanning thousands of files on every run.
# If you change the preprocessing pipeline, delete these lines and uncomment the
# compute_mean_std() block below to recalculate.
DATASET_MEAN = [0.1919, 0.1919, 0.1919]
DATASET_STD  = [0.2058, 0.2058, 0.2058]
print("Using dataset mean:", DATASET_MEAN)
print("Using dataset std :", DATASET_STD)

# --- Uncomment below to recompute from scratch ---
# def compute_mean_std(paths, sample_size=2000):
#     if len(paths) > sample_size:
#         paths = random.sample(list(paths), sample_size)
#     pixel_sum = np.zeros(3)
#     pixel_sq_sum = np.zeros(3)
#     n_pixels = 0
#     for p in tqdm(paths, desc="Computing mean/std"):
#         arr = np.asarray(Image.open(p).convert('RGB'), dtype=np.float64) / 255.0
#         pixel_sum += arr.sum(axis=(0, 1))
#         pixel_sq_sum += (arr ** 2).sum(axis=(0, 1))
#         n_pixels += arr.shape[0] * arr.shape[1]
#     mean = pixel_sum / n_pixels
#     std = np.sqrt(pixel_sq_sum / n_pixels - mean ** 2)
#     return mean.tolist(), std.tolist()
# DATASET_MEAN, DATASET_STD = compute_mean_std(df['processed_path'].tolist())
# print("Dataset mean:", DATASET_MEAN)
# print("Dataset std :", DATASET_STD)
"""
    cells[7] = make_cell(lines(norm_code))

    # ---------------------------------------------------------------
    # CELL 11: Upgraded augmentation + larger batch size
    # ---------------------------------------------------------------
    aug_code = """\
IMG_SIZE   = 224
BATCH_SIZE = 64   # increased from 16 -> 64 for faster throughput on T4

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.90, 1.10), shear=5),
    transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize(DATASET_MEAN, DATASET_STD),
    transforms.RandomErasing(p=0.20, scale=(0.02, 0.10)),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(DATASET_MEAN, DATASET_STD),
])

# TTA (test-time augmentation) transform for evaluation
tta_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(DATASET_MEAN, DATASET_STD),
])

class BrainTumorDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['processed_path']).convert('RGB')
        label = CLASS_TO_IDX[row['label']]
        if self.transform:
            img = self.transform(img)
        return img, label

train_ds = BrainTumorDataset(train_df, train_transform)
val_ds   = BrainTumorDataset(val_df,   eval_transform)
test_ds  = BrainTumorDataset(test_df,  eval_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True,
                          prefetch_factor=2)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2)

print("Batches — train:", len(train_loader), "| val:", len(val_loader), "| test:", len(test_loader))
"""
    cells[11] = make_cell(lines(aug_code))

    # ---------------------------------------------------------------
    # CELL 16 + 17: Improved training setup — MixUp, OneCycleLR, SWA
    # ---------------------------------------------------------------
    train_setup_code = """\
from sklearn.utils.class_weight import compute_class_weight
import torch.optim.swa_utils as swa_utils

y_train_idx = train_df['label'].map(CLASS_TO_IDX).values
class_weights = compute_class_weight('balanced', classes=np.arange(len(CLASS_NAMES)), y=y_train_idx)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", dict(zip(CLASS_NAMES, class_weights.round(3))))

# ---- MixUp helper ----
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
"""
    cells[16] = make_cell(lines(train_setup_code))

    # Training loop cell
    training_loop_code = """\
DRIVE_SAVE_DIR = '/content/drive/MyDrive/pdscnn_standalone_v2'
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

BEST_CKPT_PATH  = os.path.join(DRIVE_SAVE_DIR, 'pdscnn_best.pth')
FINAL_CKPT_PATH = os.path.join(DRIVE_SAVE_DIR, 'pdscnn_final.pth')
SWA_CKPT_PATH   = os.path.join(DRIVE_SAVE_DIR, 'pdscnn_swa.pth')
HISTORY_PATH    = os.path.join(DRIVE_SAVE_DIR, 'training_history.csv')
CONFIG_PATH     = os.path.join(DRIVE_SAVE_DIR, 'config.json')

EPOCHS         = 100
LR             = 3e-3    # higher LR works well with OneCycleLR
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTHING = 0.10
MIXUP_ALPHA    = 0.4
SWA_START      = 75      # start SWA after 75 epochs for final accuracy boost

optimizer = torch.optim.AdamW(pdscnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=LR,
    steps_per_epoch=len(train_loader),
    epochs=SWA_START,          # OneCycle for first SWA_START epochs
    pct_start=0.10,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1e4,
)
criterion      = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=LABEL_SMOOTHING)
scaler         = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

# SWA (Stochastic Weight Averaging) for better generalisation
swa_model      = swa_utils.AveragedModel(pdscnn)
swa_scheduler  = swa_utils.SWALR(optimizer, swa_lr=5e-5, anneal_epochs=5)

history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
best_val_acc = 0.0

for epoch in range(EPOCHS):
    pdscnn.train()
    running_loss, running_correct, n_seen = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
    use_scheduler_step = epoch < SWA_START

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        # MixUp
        mixed_imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=MIXUP_ALPHA)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            logits, _ = pdscnn(mixed_imgs)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(pdscnn.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        if use_scheduler_step:
            scheduler.step()

        running_loss    += loss.item() * imgs.size(0)
        # For accuracy, use the un-mixed labels (original) for a meaningful metric
        with torch.no_grad():
            clean_logits, _ = pdscnn(imgs)
        running_correct += (clean_logits.argmax(1) == labels).sum().item()
        n_seen          += imgs.size(0)
        pbar.set_postfix(loss=running_loss / n_seen, acc=running_correct / n_seen)

    train_loss = running_loss / n_seen
    train_acc  = running_correct / n_seen

    # SWA update
    if epoch >= SWA_START:
        swa_model.update_parameters(pdscnn)
        swa_scheduler.step()

    # Validation (no MixUp)
    pdscnn.eval()
    val_loss, val_correct, n_val = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
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

# Update SWA BN stats and save SWA model
torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
torch.save(swa_model.state_dict(), SWA_CKPT_PATH)

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
print(f"Checkpoints saved to: {DRIVE_SAVE_DIR}")
"""
    cells[17] = make_cell(lines(training_loop_code))

    nb['cells'] = cells

    with open('pdscnn_base_paper.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

if __name__ == '__main__':
    patch_notebook()
    print("pdscnn_base_paper.ipynb patched successfully.")
