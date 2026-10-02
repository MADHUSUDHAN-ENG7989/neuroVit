import json

def create_vit_notebook():
    # Load the base pdscnn notebook to extract data loading and preprocessing cells
    with open('pdscnn_base_paper.ipynb', 'r', encoding='utf-8') as f:
        pdscnn_nb = json.load(f)
    
    cells = []
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["# ViT Fine-Tuning for Brain Tumor Classification\n", "\n", "This notebook fine-tunes `google/vit-base-patch16-224-in21k` on the brain tumor dataset. Data loading and preprocessing are identical to the PDSCNN pipeline."]
    })
    
    # We will copy the cells from pdscnn_nb until the "7. PDSCNN Model" section or similar.
    copy_cells = []
    for cell in pdscnn_nb['cells']:
        if cell['cell_type'] == 'markdown':
            src = "".join(cell['source'])
            if '7. PDSCNN Model' in src or '## 7. PDSCNN' in src:
                break
        copy_cells.append(cell)
    
    cells.extend(copy_cells)
    
    # Add ViT Setup & Training cells
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 7. ViT Setup and Fine-tuning\n", "\n", "We will use `transformers` to fine-tune `google/vit-base-patch16-224-in21k`."]
    })
    
    vit_setup_code = """!pip install -q transformers
import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# ViT Configuration
model_name = 'google/vit-base-patch16-224-in21k'
num_classes = len(CLASS_NAMES)

# Load pre-trained ViT
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=num_classes,
    id2label={str(i): c for i, c in enumerate(CLASS_NAMES)},
    label2id={c: str(i) for i, c in enumerate(CLASS_NAMES)}
)

model = model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"ViT parameters: {n_params:,}")
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in vit_setup_code.split('\n')]
    })
    
    vit_train_code = """import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

y_train_idx = train_df['label'].map(CLASS_TO_IDX).values
class_weights = compute_class_weight('balanced', classes=np.arange(len(CLASS_NAMES)), y=y_train_idx)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Class weights:", dict(zip(CLASS_NAMES, class_weights.round(3))))

EPOCHS = 50
LR = 2e-5  # Lower LR for fine-tuning transformers
WEIGHT_DECAY = 0.01

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0

DRIVE_SAVE_DIR = '/content/drive/MyDrive/vit_finetuned'
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)
BEST_CKPT_PATH = os.path.join(DRIVE_SAVE_DIR, 'vit_best.pth')

for epoch in range(EPOCHS):
    model.train()
    running_loss, running_correct, n_seen = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]")
    
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(pixel_values=imgs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        running_correct += (logits.argmax(1) == labels).sum().item()
        n_seen += imgs.size(0)
        pbar.set_postfix(loss=running_loss / n_seen, acc=running_correct / n_seen)

    train_loss = running_loss / n_seen
    train_acc = running_correct / n_seen

    # Validation
    model.eval()
    val_loss, val_correct, n_val = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(pixel_values=imgs)
                logits = outputs.logits
                loss = criterion(logits, labels)
                
            val_loss += loss.item() * imgs.size(0)
            val_correct += (logits.argmax(1) == labels).sum().item()
            n_val += imgs.size(0)

    val_loss /= n_val
    val_acc = val_correct / n_val
    scheduler.step()

    print(f"Epoch {epoch+1}/{EPOCHS} — train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
          f"| val_loss {val_loss:.4f} val_acc {val_acc:.4f} | lr {optimizer.param_groups[0]['lr']:.2e}")

    history['epoch'].append(epoch + 1)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), BEST_CKPT_PATH)
        print(f"  -> New best val_acc {val_acc:.4f}, checkpoint saved!")

print(f"Training complete. Best Validation Accuracy: {best_val_acc:.4f}")
pd.DataFrame(history).to_csv(os.path.join(DRIVE_SAVE_DIR, 'vit_history.csv'), index=False)
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in vit_train_code.split('\n')]
    })
    
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 8. Test Set Evaluation"]
    })
    
    test_eval_code = """# Load best model for evaluation
model.load_state_dict(torch.load(BEST_CKPT_PATH))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Evaluating on test set"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(pixel_values=imgs)
        preds = outputs.logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = accuracy_score(all_labels, all_preds)
print(f"\\nTest Accuracy: {test_acc:.4f}")
print("\\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
"""
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in test_eval_code.split('\n')]
    })
    
    nb = {
        "cells": cells,
        "metadata": pdscnn_nb.get('metadata', {}),
        "nbformat": pdscnn_nb.get('nbformat', 4),
        "nbformat_minor": pdscnn_nb.get('nbformat_minor', 5)
    }
    
    with open('vit_finetune.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

if __name__ == '__main__':
    create_vit_notebook()
    print("vit_finetune.ipynb created successfully.")
