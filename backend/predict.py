import argparse
import json
import os
import sys
import pickle

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from transformers import ViTForImageClassification
import shap

# Note: The grad-cam package uses `pytorch_grad_cam`
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- Configuration ---
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Transforms ---
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            cl = self.clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            return Image.fromarray(img_clahe)
        else:
            cl = self.clahe.apply(img_np)
            return Image.fromarray(cl)

eval_transform = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

def denormalize(img_tensor):
    img = img_tensor.clone().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)

# --- Model Architectures ---

# 1. PDSCNN
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class PDSCNN(nn.Module):
    def __init__(self, num_classes=4, in_ch=3):
        super().__init__()
        self.branch_11 = DepthwiseSeparableConv2d(in_ch, 64, 11, padding=5)
        self.branch_9 = DepthwiseSeparableConv2d(in_ch, 64, 9, padding=4)
        self.branch_7 = DepthwiseSeparableConv2d(in_ch, 64, 7, padding=3)
        self.branch_5 = DepthwiseSeparableConv2d(in_ch, 64, 5, padding=2)
        self.branch_3 = DepthwiseSeparableConv2d(in_ch, 64, 3, padding=1)
        self.conv6 = DepthwiseSeparableConv2d(64 * 5, 256, 3, padding=0)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(2, 2)
        self.conv7 = DepthwiseSeparableConv2d(256, 128, 3, padding=0)
        self.bn7 = nn.BatchNorm2d(128)
        self.pool7 = nn.MaxPool2d(2, 2)
        self.conv8 = DepthwiseSeparableConv2d(128, 64, 3, padding=0)
        self.bn8 = nn.BatchNorm2d(64)
        self.pool8 = nn.MaxPool2d(2, 2)
        self.conv9 = DepthwiseSeparableConv2d(64, 32, 3, padding=0)
        self.bn9 = nn.BatchNorm2d(32)
        self.pool9 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward_features(self, x):
        x = torch.cat([self.branch_11(x), self.branch_9(x), self.branch_7(x), self.branch_5(x), self.branch_3(x)], dim=1)
        x = self.pool6(self.bn6(self.relu(self.conv6(x))))
        x = self.pool7(self.bn7(self.relu(self.conv7(x))))
        x = self.pool8(self.bn8(self.relu(self.conv8(x))))
        x = self.pool9(self.bn9(self.relu(self.conv9(x))))
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return x

    def forward(self, x):
        feat = self.forward_features(x)  # 512-dim for fusion
        x_cls = self.dropout(feat)
        x_cls = self.fc2(x_cls)
        x_cls = self.dropout(x_cls)
        return self.classifier(x_cls), feat


# RRELM definition so pickle can unpickle it
# RRELM definition so pickle can unpickle it
class RRELM:
    def __init__(self, n_hidden=512, C=1.0, activation='sigmoid', random_state=42):
        self.n_hidden = n_hidden
        self.C = C
        self.activation = activation
        self.rng = np.random.RandomState(random_state)
    def _activate(self, H):
        return 1.0 / (1.0 + np.exp(-H))
    def fit(self, X, y_onehot):
        pass # Not needed for inference
    def predict_proba(self, X):
        H = self._activate(X @ self.W + self.b)
        return H @ self.beta
    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# Wrapper for Grad-CAM
class PDSCNNLogitsOnly(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        logits, _ = self.base(x)
        return logits

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    args = parser.parse_args()

    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
    vit_path = os.path.join(model_dir, 'vit_finetuned/vit_best.pth')
    pdscnn_path = os.path.join(model_dir, 'pdscnn_standalone_v3/pdscnn_final.pth')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    rrelm_path = os.path.join(model_dir, 'rrelm.pkl')

    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found at {args.image_path}"}))
        sys.exit(1)

    if not os.path.exists(rrelm_path) or not os.path.exists(scaler_path):
        print(json.dumps({"error": "rrelm.pkl or scaler.pkl not found in model/ dir. Please run the provided Colab code to save them and put them there."}))
        sys.exit(1)

    print("Loading ViT...", file=sys.stderr)
    # 1. Load Models
    vit_base = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True
    ).to(device)
    
    print("Loading ViT weights...", file=sys.stderr)
    vit_sd = torch.load(vit_path, map_location=device)
    # Fix for transformers version differences (vit.layers vs vit.encoder.layer)
    new_vit_sd = {}
    for k, v in vit_sd.items():
        new_k = k.replace('vit.layers.', 'vit.encoder.layer.')
        new_k = new_k.replace('.attention.q_proj.', '.attention.attention.query.')
        new_k = new_k.replace('.attention.k_proj.', '.attention.attention.key.')
        new_k = new_k.replace('.attention.v_proj.', '.attention.attention.value.')
        new_k = new_k.replace('.attention.o_proj.', '.attention.output.dense.')
        new_k = new_k.replace('.mlp.fc1.', '.intermediate.dense.')
        new_k = new_k.replace('.mlp.fc2.', '.output.dense.')
        new_vit_sd[new_k] = v
        
    vit_base.load_state_dict(new_vit_sd, strict=False)
    vit_base.eval()

    print("Loading PDSCNN...", file=sys.stderr)
    pdscnn_model = PDSCNN(num_classes=len(CLASS_NAMES)).to(device)
    pdscnn_ckpt = torch.load(pdscnn_path, map_location=device)
    pdscnn_sd = pdscnn_ckpt.get('model_state_dict', pdscnn_ckpt) if isinstance(pdscnn_ckpt, dict) else pdscnn_ckpt
    if any(k.startswith('module.') for k in pdscnn_sd):
        pdscnn_sd = {k[len('module.'):]: v for k, v in pdscnn_sd.items() if k.startswith('module.')}
    pdscnn_model.load_state_dict(pdscnn_sd, strict=True)
    pdscnn_model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(rrelm_path, 'rb') as f:
        rrelm_model = pickle.load(f)

    print("Preprocessing image...", file=sys.stderr)
    # 2. Prepare Image
    orig_img = Image.open(args.image_path).convert('RGB')
    input_tensor = eval_transform(orig_img).unsqueeze(0).to(device)

    # 3. Prediction Pipeline
    with torch.no_grad():
        # Get ViT feature (CLS token)
        vit_out = vit_base.vit(pixel_values=input_tensor)
        vit_feat = vit_out.last_hidden_state[:, 0, :]
        
        # Get PDSCNN feature
        _, pdscnn_feat = pdscnn_model(input_tensor)

        # Fusion
        fused = torch.cat((pdscnn_feat, vit_feat), dim=1)
        
        # Scaling
        fused_np = fused.cpu().numpy()
        fused_scaled = scaler.transform(fused_np)

    raw_scores = rrelm_model.predict_proba(fused_scaled)[0]
    
    # RRELM outputs approximate one-hot vectors via least squares, which can be > 1 or < 0.
    # We apply softmax to convert them into proper confidence probabilities (0 to 1).
    exp_scores = np.exp(raw_scores - np.max(raw_scores)) # Subtract max for stability
    probs = exp_scores / np.sum(exp_scores)

    pred_idx = np.argmax(probs)
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # Base paths for output
    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    out_dir = os.path.dirname(args.image_path)
    gradcam_path = os.path.join(out_dir, f"{base_name}_gradcam.jpg")
    shap_path = os.path.join(out_dir, f"{base_name}_shap.jpg")

    rgb_img = denormalize(input_tensor.squeeze(0))

    print("Generating Grad-CAM...", file=sys.stderr)
    # 4. Grad-CAM on PDSCNN
    try:
        pdscnn_wrapped = PDSCNNLogitsOnly(pdscnn_model).to(device)
        target_layers = [pdscnn_model.conv9.pointwise]
        
        cam = GradCAM(model=pdscnn_wrapped, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        # cam_image is already uint8 (0-255)
        cv2.imwrite(gradcam_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print(f"Grad-CAM error: {e}", file=sys.stderr)
        gradcam_path = None

    print("Generating SHAP explanation...", file=sys.stderr)
    # 5. SHAP / Integrated Gradients attribution on PDSCNN
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # Integrated Gradients attribution for predicted class
        steps = 8
        baseline = torch.zeros_like(input_tensor)
        interpolated = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(1, steps + 1)]
        grads = []
        for x_step in interpolated:
            x_step = x_step.clone().detach().requires_grad_(True)
            out_step = pdscnn_wrapped(x_step)
            score = out_step[0, pred_idx]
            g = torch.autograd.grad(score, x_step)[0]
            grads.append(g)
        avg_grads = torch.mean(torch.stack(grads), dim=0)
        attributions = (input_tensor - baseline) * avg_grads  # (1, 3, 224, 224)

        # Convert to (1, 224, 224, 3) for shap.image_plot
        attr_np = attributions.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        shap_numpy = np.expand_dims(attr_np, axis=0)

        shap.image_plot([shap_numpy], np.expand_dims(rgb_img, axis=0), show=False)
        fig = plt.gcf()
        fig.set_size_inches(6, 3)
        plt.savefig(shap_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
    except Exception as e:
        print(f"SHAP error: {e}", file=sys.stderr)
        shap_path = None

    # Output JSON response for the Node server
    result = {
        "prediction": pred_class,
        "confidence": confidence,
        "gradcam": f"uploads/{os.path.basename(gradcam_path)}" if gradcam_path else None,
        "shap": f"uploads/{os.path.basename(shap_path)}" if shap_path else None
    }
    
    print(json.dumps(result))

if __name__ == "__main__":
    main()
