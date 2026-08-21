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
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride,
                                    padding=padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class ParallelDSBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        branch_ch = out_ch // 2
        self.branch_a = DepthwiseSeparableConv(in_ch, branch_ch, kernel_size=3, stride=stride)
        self.branch_b = DepthwiseSeparableConv(in_ch, branch_ch, kernel_size=5, stride=stride)
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        a = self.branch_a(x)
        b = self.branch_b(x)
        x = torch.cat([a, b], dim=1)
        return self.fuse(x)

class PDSCNN(nn.Module):
    def __init__(self, num_classes=4, in_ch=3, widths=(32, 64, 128, 256, 512)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, widths[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        stages = []
        in_c = widths[0]
        for w in widths[1:]:
            stages.append(ParallelDSBlock(in_c, w, stride=2))
            in_c = w
        self.stages = nn.Sequential(*stages)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = widths[-1]
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.gap(x).flatten(1)

    def forward(self, x):
        feat = self.forward_features(x)
        return self.classifier(feat), feat

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
    vit_path = os.path.join(model_dir, 'vit_finetuned_best.pth')
    pdscnn_path = os.path.join(model_dir, 'pdscnn_clahe_best.pth')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    rrelm_path = os.path.join(model_dir, 'rrelm.pkl')

    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found at {args.image_path}"}))
        sys.exit(1)

    if not os.path.exists(rrelm_path) or not os.path.exists(scaler_path):
        print(json.dumps({"error": "rrelm.pkl or scaler.pkl not found in model/ dir. Please run the provided Colab code to save them and put them there."}))
        sys.exit(1)

    # 1. Load Models
    vit_base = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True
    ).to(device)
    
    vit_sd = torch.load(vit_path, map_location=device)
    # Fix for transformers version differences (vit.layers vs vit.encoder.layer)
    vit_sd = {k.replace('vit.layers.', 'vit.encoder.layer.'): v for k, v in vit_sd.items()}
    vit_base.load_state_dict(vit_sd, strict=False)
    vit_base.eval()

    pdscnn_model = PDSCNN(num_classes=len(CLASS_NAMES)).to(device)
    pdscnn_model.load_state_dict(torch.load(pdscnn_path, map_location=device))
    pdscnn_model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(rrelm_path, 'rb') as f:
        rrelm_model = pickle.load(f)

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

        # Combine
        fused = torch.cat([vit_feat, pdscnn_feat], dim=1).cpu().numpy()

    # Scale and predict
    fused_scaled = scaler.transform(fused)
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

    # 4. Grad-CAM on PDSCNN
    try:
        pdscnn_wrapped = PDSCNNLogitsOnly(pdscnn_model).to(device)
        target_layers = [pdscnn_model.stages[-1].fuse[0]]
        
        cam = GradCAM(model=pdscnn_wrapped, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
        
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cv2.imwrite(gradcam_path, cv2.cvtColor(cam_image * 255, cv2.COLOR_RGB2BGR))
    except Exception as e:
        gradcam_path = None

    # 5. SHAP on PDSCNN
    try:
        # Create a dummy background of zeros (ideal background would be sample images, but zero works for quick visualization)
        background = torch.zeros((5, 3, IMG_SIZE, IMG_SIZE)).to(device)
        explainer = shap.GradientExplainer(pdscnn_wrapped, background)
        
        shap_values, _ = explainer.shap_values(input_tensor, ranked_outputs=1)
        
        if isinstance(shap_values, list):
            shap_numpy = np.asarray(shap_values[0])
        else:
            shap_numpy = np.asarray(shap_values)
            
        if shap_numpy.ndim == 5:
            shap_numpy = np.squeeze(shap_numpy, axis=-1)
        if shap_numpy.ndim == 4 and shap_numpy.shape[1] in [1, 3]:
            shap_numpy = np.transpose(shap_numpy, (0, 2, 3, 1))

        import matplotlib.pyplot as plt
        shap.image_plot([shap_numpy], np.expand_dims(rgb_img, axis=0), show=False)
        fig = plt.gcf()
        fig.set_size_inches(6, 3)
        plt.savefig(shap_path, bbox_inches='tight', dpi=100)
        plt.close(fig)
        
    except Exception as e:
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
