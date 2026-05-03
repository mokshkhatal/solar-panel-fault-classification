import argparse
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_transforms() -> Dict[str, transforms.Compose]:
    # Keep preprocessing consistent with prediction:
    # resize to 224x224 and use 3-channel RGB normalization.
    return {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        ),
    }


def build_loaders(processed_root: Path, batch_size: int) -> Tuple[Dict[str, DataLoader], List[str]]:
    tfms = get_transforms()
    datasets_map = {
        split: datasets.ImageFolder(processed_root / split, transform=tfms[split])
        for split in ["train", "val", "test"]
    }

    loaders = {
        "train": DataLoader(datasets_map["train"], batch_size=batch_size, shuffle=True),
        "val": DataLoader(datasets_map["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets_map["test"], batch_size=batch_size, shuffle=False),
    }
    class_names = datasets_map["train"].classes
    return loaders, class_names


def create_model(arch: str, num_classes: int, freeze_backbone: bool) -> nn.Module:
    if arch == "resnet18":
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            model = models.resnet18(weights=None)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif arch == "efficientnet_b0":
        try:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        except Exception:
            model = models.efficientnet_b0(weights=None)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif arch == "densenet121":
        try:
            model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        except Exception:
            model = models.densenet121(weights=None)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            if train:
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = running_corrects / max(1, total)
    return epoch_loss, epoch_acc


def evaluate_test(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Ensemble on solar module thermal faults.")
    parser.add_argument("--processed-root", type=str, default="data/processed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze pretrained backbone and train only final layer.",
    )
    parser.add_argument("--models-dir", type=str, default="models/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_root = Path(args.processed_root)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    loaders, class_names = build_loaders(processed_root, args.batch_size)
    print(f"Classes: {class_names}")
    print(f"Training on {device} for {args.epochs} epochs")
    
    architectures = ["resnet18", "efficientnet_b0", "densenet121"]

    for arch in architectures:
        print(f"\n{'='*40}")
        print(f"Training architecture: {arch}")
        print(f"{'='*40}")

        model = create_model(arch=arch, num_classes=len(class_names), freeze_backbone=args.freeze_backbone).to(device)

        criterion = nn.CrossEntropyLoss()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=args.lr)

        best_val_acc = 0.0
        best_state = None

        for epoch in range(args.epochs):
            train_loss, train_acc = run_epoch(
                model, loaders["train"], criterion, optimizer, device, train=True
            )
            val_loss, val_acc = run_epoch(
                model, loaders["val"], criterion, optimizer, device, train=False
            )

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)

        test_acc = evaluate_test(model, loaders["test"], device)
        print(f"\n{arch} Test Accuracy: {test_acc:.4f}")

        checkpoint = {
            "arch": arch,
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "best_val_acc": best_val_acc,
        }
        
        model_out = models_dir / f"{arch}_model.pth"
        torch.save(checkpoint, model_out)
        print(f"Saved model: {model_out.resolve()}")


if __name__ == "__main__":
    main()
