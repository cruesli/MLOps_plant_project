from pathlib import Path

import torch
import typer
from model import Model

from data import MyDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(model_checkpoint: Path, batch_size: int = 32) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    state = torch.load(model_checkpoint)
    model = Model()
    model.to(DEVICE)
    model.load_state_dict(state)
    dataset = MyDataset("data")
    _, test_set = dataset.load_corrupt_mnist()

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")
    print("Evaluation complete")


if __name__ == "__main__":
    typer.run(evaluate)
