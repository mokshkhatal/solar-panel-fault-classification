import torch
import torch.nn as nn
from pathlib import Path
from train import build_loaders, create_model

def evaluate_ensemble():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, class_names = build_loaders(Path("data/processed"), 32)
    test_loader = loaders["test"]
    
    architectures = ["resnet18", "efficientnet_b0", "densenet121"]
    models = []
    
    for arch in architectures:
        checkpoint = torch.load(f"models/{arch}_model.pth", map_location=device, weights_only=False)
        model = create_model(arch, len(class_names), False).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
        
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            ensemble_probs = []
            for model in models:
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                ensemble_probs.append(probs)
                
            avg_probs = torch.mean(torch.stack(ensemble_probs), dim=0)
            _, preds = torch.max(avg_probs, dim=1)
            
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
            
    accuracy = correct / max(1, total)
    print(f"Ensemble Test Accuracy: {accuracy:.4f} ({correct}/{total} correct)")

if __name__ == "__main__":
    evaluate_ensemble()
