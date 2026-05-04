import argparse
import math
from pathlib import Path
from typing import List, Tuple
import joblib

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from preprocess import get_image_info, load_and_preprocess, preprocess_for_model
from train import create_model

def estimate_severity(image_path: Path) -> Tuple[str, float]:
    """
    Estimate fault severity from the difference between the 99th percentile
    intensity (hotspot) and the median intensity (ambient panel temperature).
    """
    pil_img = load_and_preprocess(image_path)
    gray = np.array(pil_img.convert("L"), dtype=np.float32)
    
    p99 = np.percentile(gray, 99)
    median = np.median(gray)
    delta = float(p99 - median)

    if delta < 30:
        level = "Low"
    elif delta < 70:
        level = "Medium"
    else:
        level = "High"

    return level, delta

def calculate_normalized_entropy(probs: np.ndarray) -> float:
    """Calculate the normalized predictive entropy of a probability distribution (numpy)."""
    num_classes = len(probs)
    if num_classes <= 1:
        return 0.0
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    max_entropy = np.log(num_classes)
    normalized_entropy = float(entropy / max_entropy)
    return normalized_entropy

def generate_gradcam(model: nn.Module, arch: str, tensor: torch.Tensor, image_path: Path, target_class: int, output_path: Path):
    """Generate and save a Grad-CAM heatmap."""
    # Determine the target layer for Grad-CAM
    if arch == "resnet18":
        target_layers = [model.layer4[-1]]
    elif arch == "efficientnet_b0":
        target_layers = [model.features[-1]]
    elif arch == "densenet121":
        target_layers = [model.features[-1]]
    else:
        print(f"Grad-CAM not configured for architecture: {arch}")
        return

    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    
    grayscale_cam = cam(input_tensor=tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Load original image and normalize to [0, 1]
    pil_img = load_and_preprocess(image_path)
    img_array = np.array(pil_img, dtype=np.float32) / 255.0
    
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)
    
    # Save the visualization
    cv2.imwrite(str(output_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM heatmap to: {output_path.resolve()}")

def predict_image(
    image_path: Path, models_dir: Path, device: torch.device
) -> Tuple[str, float, float, str, float]:
    
    architectures = ["resnet18", "efficientnet_b0", "densenet121"]
    backbones = []
    svms = []
    class_names = []

    # Prepare input tensor
    tensor = preprocess_for_model(image_path, device=device)

    for arch in architectures:
        model_path = models_dir / f"{arch}_model.pth"
        svm_path = models_dir / f"{arch}_svm.joblib"
        
        if not model_path.exists() or not svm_path.exists():
            raise FileNotFoundError(f"Components for {arch} not found in {models_dir}")
            
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if not class_names:
            class_names = checkpoint["class_names"]
            
        # Load the PyTorch model for feature extraction
        model = create_model(arch=arch, num_classes=len(class_names), freeze_backbone=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(device)
        
        # 1. Prediction using SVM
        svm_clf = joblib.load(svm_path)
        
        # Clone model to modify it for feature extraction without affecting Grad-CAM (which needs the head)
        feat_extractor = create_model(arch=arch, num_classes=len(class_names), freeze_backbone=False)
        feat_extractor.load_state_dict(checkpoint["model_state_dict"])
        if arch == "resnet18":
            feat_extractor.fc = nn.Identity()
        elif arch == "efficientnet_b0":
            feat_extractor.classifier[1] = nn.Identity()
        elif arch == "densenet121":
            feat_extractor.classifier = nn.Identity()
        
        feat_extractor.to(device)
        feat_extractor.eval()
        
        with torch.no_grad():
            features = feat_extractor(tensor).cpu().numpy()
            probs = svm_clf.predict_proba(features) # [1, num_classes]
            svms.append(probs[0])
            
        backbones.append((arch, model))

    # Weighted ensembling weights
    # EfficientNet-B0 is significantly stronger, so we give it 70% weight
    weights = {
        "efficientnet_b0": 0.70,
        "densenet121": 0.15,
        "resnet18": 0.15
    }
    
    # Calculate weighted average of probabilities
    weighted_probs = []
    for arch, probs in zip(architectures, svms):
        weighted_probs.append(probs * weights[arch])
        
    avg_probs = np.sum(weighted_probs, axis=0)
    idx = np.argmax(avg_probs)
    confidence = avg_probs[idx]
    
    pred_class = class_names[idx]
    
    # Uncertainty quantification
    entropy = calculate_normalized_entropy(avg_probs)
    
    # Severity calculation
    severity_level, delta_value = estimate_severity(image_path)
    
    # Generate Grad-CAM using the first model (ResNet18 usually)
    # We use the original model with head for Grad-CAM gradients
    first_arch, first_model = backbones[0]
    heatmap_path = Path("prediction_heatmap.jpg")
    generate_gradcam(first_model, first_arch, tensor, image_path, int(idx), heatmap_path)

    return pred_class, float(confidence), entropy, severity_level, delta_value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict solar panel fault from a thermal image using a CNN+SVM ensemble."
    )
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models/",
        help="Path to directory containing trained models",
    )
    parser.add_argument(
        "--show-image-info",
        action="store_true",
        help="Print raw image diagnostics before prediction",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = Path(args.image_path)
    models_dir = Path(args.models_dir)

    if args.show_image_info:
        from preprocess import get_image_info
        info = get_image_info(image_path)
        print("\n-- Raw image diagnostics ------------------------------")
        for k, v in info.items():
            print(f"  {k:25s}: {v}")
        print("----------------------------------------------------\n")

    pred_class, confidence, entropy, severity_level, delta_value = predict_image(
        image_path=image_path, models_dir=models_dir, device=device
    )

    print(f"Predicted Class : {pred_class}")
    print(f"Confidence      : {confidence:.4f} (SVM Ensemble Soft Voting)")
    print(f"Norm. Entropy   : {entropy:.4f} (0 = Certain, 1 = Uncertain)")
    print(f"Severity Level  : {severity_level}")
    print(f"Thermal Delta   : {delta_value:.2f} (99th percentile - median)")
    print(
        "\nNote: Severity is based on image-intensity anomalies. "
        "It does not represent real temperature."
    )


if __name__ == "__main__":
    main()
