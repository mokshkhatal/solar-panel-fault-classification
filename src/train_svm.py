import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
from train import build_loaders, create_model

def extract_features(model, loader, device):
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            # Forward pass
            feat = model(imgs)
            features.append(feat.cpu().numpy())
            labels.append(lbls.numpy())
            
    return np.concatenate(features), np.concatenate(labels)

def train_svms():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_root = Path("data/processed")
    models_dir = Path("models")
    
    loaders, class_names = build_loaders(processed_root, 32)
    architectures = ["resnet18", "efficientnet_b0", "densenet121"]
    
    for arch in architectures:
        print(f"\nProcessing {arch}...")
        
        # 1. Load the PyTorch model
        checkpoint = torch.load(models_dir / f"{arch}_model.pth", map_location=device)
        model = create_model(arch, len(class_names), False)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # 2. Convert to feature extractor by replacing classification layer with Identity
        if arch == "resnet18":
            model.fc = nn.Identity()
        elif arch == "efficientnet_b0":
            model.classifier[1] = nn.Identity()
        elif arch == "densenet121":
            model.classifier = nn.Identity()
            
        model = model.to(device)
        
        # 3. Extract features for train and val
        print(f"Extracting training features for {arch}...")
        X_train, y_train = extract_features(model, loaders["train"], device)
        
        print(f"Extracting validation features for {arch}...")
        X_val, y_val = extract_features(model, loaders["val"], device)
        
        # 4. Train SVM
        print(f"Training SVM for {arch}...")
        # Use a pipeline with StandardScaler and SVC with probability=True for ensembling
        clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
        clf.fit(X_train, y_train)
        
        # 5. Evaluate on validation set
        val_acc = clf.score(X_val, y_val)
        print(f"{arch} SVM Validation Accuracy: {val_acc:.4f}")
        
        # 6. Save SVM
        svm_path = models_dir / f"{arch}_svm.joblib"
        joblib.dump(clf, svm_path)
        print(f"Saved SVM to {svm_path}")

if __name__ == "__main__":
    train_svms()
