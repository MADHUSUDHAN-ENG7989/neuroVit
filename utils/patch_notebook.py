import json
import re

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update IMG_SIZE in cell 30
cell_30 = nb['cells'][30]
source_30 = ''.join(cell_30['source'])
source_30 = source_30.replace('IMG_SIZE = 224', 'IMG_SIZE = 124')
cell_30['source'] = [line + '\n' for line in source_30.split('\n')]
if cell_30['source'][-1].endswith('\n\n'):
    cell_30['source'][-1] = cell_30['source'][-1].strip('\n')

# Replace model definition in cell 33
new_model_code = """class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        # Note: BN is applied after the pointwise conv in the sequential block in our wrapper, 
        # but for the parallel branches we just return the raw conv output to concatenate
        return x

class PDSCNN(nn.Module):
    def __init__(self, num_classes=4, in_ch=3):
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
        self.flatten = nn.Flatten()
        
        # FC
        self.fc1 = nn.Linear(400, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        
        # Classifier (to support original notebook's training loop)
        self.classifier = nn.Linear(256, num_classes)
        
    def forward_features(self, x):
        # Parallel stage
        b11 = self.branch_11(x)
        b9  = self.branch_9(x)
        b7  = self.branch_7(x)
        b5  = self.branch_5(x)
        b3  = self.branch_3(x)
        
        x = torch.cat([b11, b9, b7, b5, b3], dim=1)
        
        x = self.pool6(self.bn6(self.relu(self.conv6(x))))
        x = self.pool7(self.bn7(self.relu(self.conv7(x))))
        x = self.pool8(self.bn8(self.relu(self.conv8(x))))
        x = self.pool9(self.bn9(self.relu(self.conv9(x))))
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        feat = self.forward_features(x)
        out = self.classifier(self.dropout(feat))
        return out, feat

pdscnn = PDSCNN(num_classes=len(CLASS_NAMES)).to(device)
n_params = sum(p.numel() for p in pdscnn.parameters())
print(f"PDSCNN parameters: {n_params:,}")
"""
nb['cells'][33]['source'] = [line + '\n' for line in new_model_code.split('\n')]
nb['cells'][33]['source'][-1] = nb['cells'][33]['source'][-1].strip('\n')

# Append RRELM implementation and feature extraction at the end
rrelm_md = {"cell_type": "markdown", "metadata": {}, "source": ["## 12. RRELM Feature Extraction and Classification"]}
rrelm_code = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + '\n' for line in """
from sklearn.preprocessing import StandardScaler

def extract_features(model, loader):
    model.eval()
    all_feats, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(device)
            _, feats = model(imgs)
            all_feats.append(feats.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)

# We use the train_loader but without shuffle/augmentations for extraction
train_loader_eval = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

X_train, y_train = extract_features(pdscnn, train_loader_eval)
X_test, y_test = extract_features(pdscnn, test_loader)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

class RRELM:
    def __init__(self, n_hidden=1500, C=10.0, activation='sigmoid', random_state=42):
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

rrelm = RRELM(n_hidden=1500, C=10.0, random_state=SEED)
rrelm.fit(X_train_s, to_onehot(y_train, 4))

test_preds = rrelm.predict(X_test_s)
acc = accuracy_score(y_test, test_preds)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, test_preds, average='macro')
print(f"RRELM Test Accuracy: {acc:.4f}")
print(f"Macro Precision: {precision:.4f} | Macro Recall: {recall:.4f} | Macro F1: {f1:.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, test_preds, target_names=CLASS_NAMES))
""".split('\n')]}
rrelm_code['source'][-1] = rrelm_code['source'][-1].strip('\n')

nb['cells'].extend([rrelm_md, rrelm_code])

with open('pdscnn_base_paper.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Saved updated notebook to pdscnn_base_paper.ipynb")
