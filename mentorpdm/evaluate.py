import argparse
from pathlib import Path

import torch

from .data_loader import PaderbornDataset, create_loaders
from .model import SimpleCNN
from .train import evaluate


def evaluate_saved(data_dir: Path, model_path: Path, batch_size: int = 32, device: str = None):
    dataset = PaderbornDataset(str(data_dir))
    _, val_loader = create_loaders(dataset, batch_size=batch_size)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = SimpleCNN(num_classes=len(dataset.label2idx)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    criterion = torch.nn.CrossEntropyLoss()
    loss, acc = evaluate(model, val_loader, device, criterion)
    print(f"Validation loss: {loss:.4f} | accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved model")
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    evaluate_saved(args.data_dir, args.model, batch_size=args.batch_size, device=args.device)
