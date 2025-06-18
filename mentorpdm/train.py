import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .data_loader import PaderbornDataset, create_loaders
from .model import SimpleCNN


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    return loss, acc


def train(root_dir: Path, epochs: int = 10, batch_size: int = 32, lr: float = 1e-3, device: str = None):
    dataset = PaderbornDataset(str(root_dir))
    train_loader, val_loader = create_loaders(dataset, batch_size=batch_size)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = SimpleCNN(num_classes=len(dataset.label2idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch:2d}: train loss {train_loss:.4f} | val loss {val_loss:.4f} | val acc {val_acc:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model on the Paderborn dataset")
    parser.add_argument("data_dir", type=Path, help="Root directory of the dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
