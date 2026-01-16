import json
from pathlib import Path

import torch
import typer
from model import Model

from data import MyDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

METADATA_PATH = Path("data/processed/metadata.json")

def evaluate(model_checkpoint: Path, batch_size: int = 32) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    num_classes = len(metadata["plant_to_idx"])

    state = torch.load(model_checkpoint)
    model = Model(
        num_classes=num_classes,
        in_channels=3,
        conv1_out=32,
        conv1_kernel=3,
        conv1_stride=1,
        conv2_out=64,
        conv2_kernel=3,
        conv2_stride=3,
        conv2_padding=1,
        dropout=0.2,
    )
    model.to(DEVICE)
    model.load_state_dict(state)
    dataset = MyDataset("data")
    _, test_set = dataset.load_plantvillage(target="plant")

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