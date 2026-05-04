import torch
import torch.nn as nn
import sys
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Add src to sys.path
sys.path.append(str(Path(__file__).parent))
from train import build_loaders, create_model

def evaluate_detailed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_names = build_loaders(Path("data/processed"), 32)
    test_loader = loaders["test"]
    
    architectures = ["resnet18", "efficientnet_b0", "densenet121"]
    
    # Load backbones (feature extractors) and SVMs
    backbones = {}
    svms = {}
    
    print("Loading models and SVM components...")
    for arch in architectures:
        # Load backbone
        checkpoint = torch.load(f"models/{arch}_model.pth", map_location=device, weights_only=False)
        model = create_model(arch, len(class_names), False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Convert to feature extractor
        if arch == "resnet18":
            model.fc = nn.Identity()
        elif arch == "efficientnet_b0":
            model.classifier[1] = nn.Identity()
        elif arch == "densenet121":
            model.classifier = nn.Identity()
            
        model = model.to(device)
        model.eval()
        backbones[arch] = model
        
        # Load SVM
        svms[arch] = joblib.load(f"models/{arch}_svm.joblib")
        
    y_true = []
    y_preds_dict = {arch: [] for arch in architectures}
    ensemble_probs_list = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            y_true.extend(labels.numpy())
            
            batch_probs = []
            for arch in architectures:
                # 1. Extract features
                feats = backbones[arch](images).cpu().numpy()
                # 2. Get SVM probabilities
                probs = svms[arch].predict_proba(feats)
                batch_probs.append(probs)
                
                # Individual predictions
                preds = np.argmax(probs, axis=1)
                y_preds_dict[arch].extend(preds)
            
            # Stack batch probabilities for ensemble [3, batch_size, num_classes]
            ensemble_probs_list.append(np.array(batch_probs))
            
    # Calculate Weighted Ensemble Predictions
    # EfficientNet-B0 (idx 1), DenseNet (idx 2), ResNet (idx 0)
    weights = np.array([0.15, 0.7, 0.15]).reshape(3, 1, 1) # [arch, 1, 1] for broadcasting
    
    # ensemble_probs_list is list of [3, batch_size, num_classes]
    # concatenate to [3, total_samples, num_classes]
    all_probs = np.concatenate(ensemble_probs_list, axis=1)
    
    # Multiply each model's probabilities by its weight
    weighted_all_probs = all_probs * weights
    avg_probs = np.sum(weighted_all_probs, axis=0)
    
    y_ensemble_preds = np.argmax(avg_probs, axis=1)
    
    print("\n" + "="*50)
    print(" INDIVIDUAL MODEL ACCURACIES ")
    print("="*50)
    for arch in architectures:
        acc = accuracy_score(y_true, y_preds_dict[arch])
        print(f"{arch:20}: {acc:.4f}")
    
    ensemble_acc = accuracy_score(y_true, y_ensemble_preds)
    print(f"{'Ensemble':20}: {ensemble_acc:.4f}")
    print("="*50)
    
    print("\n" + "="*50)
    print(" CLASS-WISE METRICS (ENSEMBLE) ")
    print("="*50)
    report = classification_report(y_true, y_ensemble_preds, target_names=class_names)
    print(report)
    print("="*50)

if __name__ == "__main__":
    evaluate_detailed()
