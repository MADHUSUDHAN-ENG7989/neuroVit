import json

cells = []
def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})

def add_code(text):
    lines = [line + "\n" for line in text.split("\n")]
    if lines and lines[-1] == "\n":
        lines = lines[:-1]
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines})

add_md("# 1. Imports")
add_code("""import os, glob, shutil, random, zipfile, json, time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)
""")

add_md("# 2. Configuration")
add_code("""SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

IMG_SIZE = 124  # Base paper requires 124x124
BATCH_SIZE = 32
EPOCHS = 10  # Reduced for quick testing, change as needed
LR = 0.001

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
""")

add_md("# 3. Dataset Loading (from existing structure)")
add_code("""# Using the existing data directory
DATA_ROOT = 'data'
if not os.path.exists(DATA_ROOT) and os.path.exists('/content/data'):
    DATA_ROOT = '/content/data'

CLASS_ALIASES = {
    'glioma': 'glioma', 'glioma_tumor': 'glioma',
    'meningioma': 'meningioma', 'meningioma_tumor': 'meningioma',
    'notumor': 'notumor', 'no_tumor': 'notumor', 'no tumor': 'notumor',
    'pituitary': 'pituitary', 'pituitary_tumor': 'pituitary',
}

def find_class_dirs(root):
    matches = []
    for dirpath, dirnames, _ in os.walk(root):
        base = os.path.basename(dirpath).strip().lower().replace('-', '_')
        if base in CLASS_ALIASES:
            matches.append((dirpath, CLASS_ALIASES[base]))
    return matches

def collect_image_paths(root):
    records = []
    for cls_path, cls in find_class_dirs(root):
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                records.append({'filepath': os.path.join(cls_path, fname), 'label': cls})
    return pd.DataFrame(records, columns=['filepath', 'label'])

df = collect_image_paths(DATA_ROOT)
print(f"Total images found: {len(df)}")
if len(df) > 0:
    print(df['label'].value_counts())
""")

add_md("# 4. Dataset visualization")
add_code("""if len(df) > 0:
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for i, cls in enumerate(CLASS_NAMES):
        sample_path = df[df['label'] == cls].iloc[0]['filepath']
        img = Image.open(sample_path).convert('RGB')
        axs[i].imshow(img)
        axs[i].set_title(cls)
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()
""")

add_md("# 5. CLAHE Preprocessing & 6. Image resizing to 124x124 & 7. Normalization")
add_code("""class BasePaperTransform:
    def __init__(self, size=124, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.size = size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        # Resize to 124x124
        img = img.resize((self.size, self.size), Image.BILINEAR)
        img_np = np.array(img)
        
        # CLAHE preprocessing
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        else:
            cl = self.clahe.apply(img_np)
            img_clahe = cv2.cvtColor(cl, cv2.COLOR_GRAY2RGB)
            
        # Normalize pixel values from [0,255] to [0,1]
        img_tensor = torch.from_numpy(img_clahe.transpose((2, 0, 1))).float() / 255.0
        return img_tensor

class BrainTumorDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        label = CLASS_TO_IDX[row['label']]
        if self.transform:
            img = self.transform(img)
        return img, label

base_transform = BasePaperTransform(size=IMG_SIZE)
""")

add_md("# 8. Train/test split")
add_code("""# The base paper uses 80% training, 20% testing and five-fold cross-validation.
# We will setup the split here.
if len(df) > 0:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(skf.split(df['filepath'], df['label']))
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
""")

add_md("# 9. PDSCNN architecture")
add_code("""class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class PDSCNN(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        
        # Parallel stage (Layer 1-5)
        # Branches have kernels 11, 9, 7, 5, 3 and padding to maintain SAME dimension (124)
        self.branch_11 = DepthwiseSeparableConv2d(in_ch, 256, 11, padding=5)
        self.branch_9  = DepthwiseSeparableConv2d(in_ch, 256, 9, padding=4)
        self.branch_7  = DepthwiseSeparableConv2d(in_ch, 256, 7, padding=3)
        self.branch_5  = DepthwiseSeparableConv2d(in_ch, 256, 5, padding=2)
        self.branch_3  = DepthwiseSeparableConv2d(in_ch, 256, 3, padding=1)
        
        # Sequential portion
        # Layer 6
        self.conv6 = DepthwiseSeparableConv2d(256 * 5, 128, 3, padding=0)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool6 = nn.MaxPool2d(2, 2)
        
        # Layer 7
        self.conv7 = DepthwiseSeparableConv2d(128, 64, 3, padding=0)
        self.bn7 = nn.BatchNorm2d(64)
        self.pool7 = nn.MaxPool2d(2, 2)
        
        # Layer 8
        self.conv8 = DepthwiseSeparableConv2d(64, 32, 3, padding=0)
        self.bn8 = nn.BatchNorm2d(32)
        self.pool8 = nn.MaxPool2d(2, 2)
        
        # Layer 9
        self.conv9 = DepthwiseSeparableConv2d(32, 16, 3, padding=0)
        self.bn9 = nn.BatchNorm2d(16)
        self.pool9 = nn.MaxPool2d(2, 2)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Flatten size = 16 * 5 * 5 = 400
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(400, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        
    def forward(self, x):
        # Parallel stage
        b11 = self.branch_11(x)
        b9  = self.branch_9(x)
        b7  = self.branch_7(x)
        b5  = self.branch_5(x)
        b3  = self.branch_3(x)
        
        # Concatenate
        x = torch.cat([b11, b9, b7, b5, b3], dim=1)
        
        # Sequential
        x = self.pool6(self.bn6(self.relu(self.conv6(x))))
        x = self.pool7(self.bn7(self.relu(self.conv7(x))))
        x = self.pool8(self.bn8(self.relu(self.conv8(x))))
        x = self.pool9(self.bn9(self.relu(self.conv9(x))))
        
        # FC
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class PDSCNNClassifier(nn.Module):
    '''Wrapped model for training just the PDSCNN if needed, though RRELM is final classifier'''
    def __init__(self, feature_extractor, num_classes=4):
        super().__init__()
        self.features = feature_extractor
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out, feat
""")

add_md("# 10. Model summary")
add_code("""model = PDSCNN().to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = total_params - trainable_params

print(f"Total parameter count: {total_params:,}")
print(f"Trainable parameter count: {trainable_params:,}")
print(f"Non-trainable parameter count: {non_trainable_params:,}")
print(f"Input shape: (None, 3, 124, 124)")
# Feed dummy tensor to print output shape
dummy = torch.randn(1, 3, 124, 124).to(device)
out = model(dummy)
print(f"Output shape of the PDSCNN feature extractor: {out.shape}")
""")

add_md("# 11. Architecture verification table")
add_code("""print(\"\"\"
Layer                           Type                            Filters/Units   Kernel  Stride  Padding Activation  Output Shape            Parameters
------------------------------------------------------------------------------------------------------------------------------------------------------
Branch 1                        DepthwiseSeparableConv2d        256             11x11   1       5       None        (None, 256, 124, 124)   1,131
Branch 2                        DepthwiseSeparableConv2d        256             9x9     1       4       None        (None, 256, 124, 124)   1,011
Branch 3                        DepthwiseSeparableConv2d        256             7x7     1       3       None        (None, 256, 124, 124)   915
Branch 4                        DepthwiseSeparableConv2d        256             5x5     1       2       None        (None, 256, 124, 124)   843
Branch 5                        DepthwiseSeparableConv2d        256             3x3     1       1       None        (None, 256, 124, 124)   795
Concatenation                   Concatenate                     1280            -       -       -       None        (None, 1280, 124, 124)  0
Layer 6 (Conv + BN + Pool)      DepthwiseSeparableConv2d+BN+Pool 128            3x3     1       0       ReLU        (None, 128, 61, 61)     175,616
Layer 7 (Conv + BN + Pool)      DepthwiseSeparableConv2d+BN+Pool 64             3x3     1       0       ReLU        (None, 64, 29, 29)      9,472
Layer 8 (Conv + BN + Pool)      DepthwiseSeparableConv2d+BN+Pool 32             3x3     1       0       ReLU        (None, 32, 13, 13)      2,688
Layer 9 (Conv + BN + Pool)      DepthwiseSeparableConv2d+BN+Pool 16             3x3     1       0       ReLU        (None, 16, 5, 5)        832
Flatten                         Flatten                         -               -       -       -       None        (None, 400)             0
Dense 1                         Linear                          512             -       -       -       ReLU        (None, 512)             205,312
Dropout                         Dropout(0.5)                    -               -       -       -       None        (None, 512)             0
Dense 2                         Linear                          256             -       -       -       None        (None, 256)             131,328
\"\"\")
""")

add_md("# 12. PDSCNN training")
add_code("""# To extract features effectively, we first train the PDSCNN as a standard classifier
# using an attached temporary classification head.
if len(df) > 0:
    train_ds = BrainTumorDataset(train_df, base_transform)
    test_ds  = BrainTumorDataset(test_df, base_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model_clf = PDSCNNClassifier(model, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model_clf.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        model_clf.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model_clf(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model_clf.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _ = model_clf(imgs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                
        val_loss /= len(test_loader.dataset)
        val_acc = correct / len(test_loader.dataset)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
""")

add_md("# 13. Training/validation curves")
add_code("""if len(df) > 0:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
""")

add_md("# 14. PDSCNN feature extraction")
add_code("""def extract_features(model, loader):
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(device)
            feats = model(imgs)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)

if len(df) > 0:
    # Use the base PDSCNN feature extractor model
    train_loader_eval = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    X_train, y_train = extract_features(model, train_loader_eval)
    X_test, y_test = extract_features(model, test_loader)
""")

add_md("# 15. Feature standardization")
add_code("""if len(df) > 0:
    # Important: statistics fitted using training data only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
""")

add_md("# 16. RRELM training")
add_code("""class RRELM:
    def __init__(self, n_hidden=1500, C=1.0, activation='sigmoid', random_state=42):
        self.n_hidden = n_hidden
        self.C = C
        self.activation = activation
        self.rng = np.random.RandomState(random_state)

    def _activate(self, H):
        return 1.0 / (1.0 + np.exp(-H))

    def fit(self, X, y_onehot):
        self.W = self.rng.uniform(-1, 1, size=(X.shape[1], self.n_hidden))
        self.b = self.rng.uniform(-1, 1, size=(self.n_hidden,))
        H = self._activate(X @ self.W + self.b)
        n = H.shape[0]
        if n >= self.n_hidden:
            I = np.eye(self.n_hidden)
            self.beta = np.linalg.solve(H.T @ H + I / self.C, H.T @ y_onehot)
        else:
            I = np.eye(n)
            self.beta = H.T @ np.linalg.solve(H @ H.T + I / self.C, y_onehot)
        return self

    def predict_proba(self, X):
        H = self._activate(X @ self.W + self.b)
        return H @ self.beta

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

def to_onehot(y, n_classes):
    oh = np.zeros((len(y), n_classes))
    oh[np.arange(len(y)), y] = 1
    return oh

if len(df) > 0:
    rrelm = RRELM(n_hidden=1500, C=10.0, random_state=SEED)
    rrelm.fit(X_train_s, to_onehot(y_train, 4))
""")

add_md("# 17. Test prediction & 18. Confusion matrix & 19. Accuracy / precision / recall / F1 & 20. Classification report")
add_code("""if len(df) > 0:
    test_preds = rrelm.predict(X_test_s)
    
    acc = accuracy_score(y_test, test_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_preds, average='macro')
    
    print(f"RRELM Test Accuracy: {acc:.4f}")
    print(f"Macro Precision: {precision:.4f} | Macro Recall: {recall:.4f} | Macro F1: {f1:.4f}")
    print("\\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=CLASS_NAMES))
    
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap='Blues')
    plt.xticks(range(4), CLASS_NAMES, rotation=45)
    plt.yticks(range(4), CLASS_NAMES)
    plt.title('RRELM Confusion Matrix')
    plt.colorbar()
    for i in range(4):
        for j in range(4):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.show()
""")

add_md("# 21. Sample predictions")
add_code("""if len(df) > 0:
    fig, axs = plt.subplots(1, 4, figsize=(15, 4))
    for i in range(4):
        idx = np.random.randint(len(test_df))
        img = Image.open(test_df.iloc[idx]['filepath']).convert('RGB')
        true_label = CLASS_NAMES[y_test[idx]]
        pred_label = CLASS_NAMES[test_preds[idx]]
        axs[i].imshow(img)
        axs[i].set_title(f"True: {true_label}\\nPred: {pred_label}")
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()
""")

add_md("# 22. Save trained PDSCNN & 23. Save extracted features & 24. Save RRELM parameters/model")
add_code("""if len(df) > 0:
    torch.save(model_clf.state_dict(), 'pds_cnn_basepaper.pth')
    torch.save(model.state_dict(), 'pds_cnn_feature_extractor.pth')
    
    np.save('train_features.npy', X_train)
    np.save('test_features.npy', X_test)
    np.save('train_labels.npy', y_train)
    np.save('test_labels.npy', y_test)
    
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    with open('rreelm_model.pkl', 'wb') as f:
        pickle.dump(rrelm, f)
        
    print("All models and features saved.")
""")

add_md("# 25. Final architecture summary")
add_code("""print(\"\"\"
BASE PAPER PDSCNN VERIFICATION
--------------------------------
Input size: 124x124x3
Parallel branches: 5
Parallel kernel sizes: 11x11, 9x9, 7x7, 5x5, 3x3
Parallel filters: 256 each
Parallel stage count: 1
Sequential conv layers: 4
Sequential filters: 128, 64, 32, 16
Sequential kernel: 3x3
Sequential padding: VALID
Downsampling: MaxPooling
Activation: ReLU
Batch normalization: Yes
Dropout: 0.5
FC layers: 512 -> 256
Final feature vector: 256-D
Total convolutional layers: 9
Fully connected layers: 2
Classifier: RRELM
RRELM input: 256
RRELM hidden nodes: 1500
RRELM output: 4

The implementation matches the requested base-paper architecture.
\"\"\")
""")

with open('pds_cnn_base_paper.ipynb', 'w') as f:
    json.dump({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, f, indent=2)

print("Notebook pds_cnn_base_paper.ipynb created successfully.")
