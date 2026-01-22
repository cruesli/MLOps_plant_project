import json
import os
from pathlib import Path

import torch
from omegaconf import DictConfig

from src.plants.model import Model
from src.plants.train import _train_model


def _ensure_dummy_processed_data() -> None:
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    if (processed_dir / "train_images.pt").exists():
        return

    dummy_train_images = torch.randn(10, 3, 28, 28)
    dummy_train_labels = torch.randint(0, 5, (10,))
    dummy_train_disease = torch.randint(0, 2, (10,))
    dummy_train_plant = torch.randint(0, 3, (10,))

    dummy_val_images = torch.randn(5, 3, 28, 28)
    dummy_val_labels = torch.randint(0, 5, (5,))
    dummy_val_disease = torch.randint(0, 2, (5,))
    dummy_val_plant = torch.randint(0, 3, (5,))

    torch.save(dummy_train_images, processed_dir / "train_images.pt")
    torch.save(dummy_train_labels, processed_dir / "train_labels.pt")
    torch.save(dummy_train_disease, processed_dir / "train_disease_labels.pt")
    torch.save(dummy_train_plant, processed_dir / "train_plant_labels.pt")

    torch.save(dummy_val_images, processed_dir / "val_images.pt")
    torch.save(dummy_val_labels, processed_dir / "val_labels.pt")
    torch.save(dummy_val_disease, processed_dir / "val_disease_labels.pt")
    torch.save(dummy_val_plant, processed_dir / "val_plant_labels.pt")

    metadata = {
        "class_to_idx": {"class1": 0, "class2": 1, "class3": 2, "class4": 3, "class5": 4},
        "disease_to_idx": {"healthy": 0, "diseased": 1},
        "plant_to_idx": {"plant1": 0, "plant2": 1, "plant3": 2},
        "mean": 0.5,
        "std": 0.5,
        "splits": {"train": "dummy_train", "val": "dummy_val"},
    }
    (processed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def get_model_state_dict(model_path, num_classes: int | None = None) -> dict:
    """Load and return the state dict of a model, deriving shapes from the checkpoint."""
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    inferred_num_classes = state_dict["fc.weight"].shape[0]
    inferred_in_channels = state_dict["layer1.0.weight"].shape[1]
    model = Model(
        num_classes=num_classes or inferred_num_classes,
        in_channels=inferred_in_channels,
        conv1_out=32,
        conv1_kernel=3,
        conv1_stride=1,
        conv2_out=64,
        conv2_kernel=3,
        conv2_stride=3,
        conv2_padding=1,
        dropout=0.2,
    )
    model.load_state_dict(state_dict)
    return model.state_dict()


def test_reproducibility(tmp_path):
    """Test the reproducibility of the training script."""
    _ensure_dummy_processed_data()

    cfg = DictConfig(
        {
            "experiments": {
                "lr": 1e-3,
                "epochs": 1,
                "batch_size": 8,
                "seed": 42,
                "device": "cpu",  # Use CPU for testing
                "metadata_path": "data/processed/metadata.json",
                "model_dir": str(tmp_path),  # Use temporary directory for model output
                "wandb": {"enabled": False, "entity": "test", "project": "test"},
                "artifact": {"name": "test_model", "type": "model", "description": "test model"},
                "target": "class",  # Assuming 'class' as the default target for the dummy data
            },
            "model": {
                "in_channels": 3,  # Dummy data is RGB
                "conv1_out": 32,
                "conv1_kernel": 3,
                "conv1_stride": 1,
                "conv2_out": 64,
                "conv2_kernel": 3,
                "conv2_stride": 3,
                "conv2_padding": 1,
                "dropout": 0.2,
            },
            "dataloader": {
                "shuffle": True,
                "num_workers": 0,
                "data_dir": "data",  # Path to dummy data
            },
        }
    )
    # Dynamically set num_classes from metadata, similar to how train does it
    with open("data/processed/metadata.json") as f:
        metadata = json.load(f)
    cfg.experiments.num_classes = len(metadata[f"{cfg.experiments.target}_to_idx"])

    # Ensure a clean slate for model output
    model_path = tmp_path / "model.pth"
    if model_path.exists():
        os.remove(model_path)

    # First run
    _train_model(cfg)  # Calls the patched train function
    state_dict_1 = get_model_state_dict(model_path)
    os.remove(model_path)

    # Second run
    _train_model(cfg)  # Calls the patched train function again
    state_dict_2 = get_model_state_dict(model_path)

    # Check that the model weights are the same
    for key in state_dict_1:
        assert torch.all(torch.eq(state_dict_1[key], state_dict_2[key]))
