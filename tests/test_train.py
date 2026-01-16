from plants.train import _train_model
from plants.model import Model
import torch
import os
import json
from pathlib import Path
from omegaconf import DictConfig
import pytest


def get_model_state_dict(model_path, num_classes):
    """Load and return the state dict of a model."""
    # Assuming these model parameters for the dummy data setup in test_data.py
    # num_classes is passed from the test, in_channels is hardcoded for the dummy data.
    model = Model(
        num_classes=num_classes,
        in_channels=1,
        conv1_out=32,
        conv1_kernel=3,
        conv1_stride=1,
        conv2_out=64,
        conv2_kernel=3,
        conv2_stride=3,
        conv2_padding=1,
        dropout=0.2,
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model.state_dict()


def test_reproducibility(tmp_path):
    """Test the reproducibility of the training script."""

    cfg = DictConfig({
        "experiments": {
            "lr": 1e-3,
            "epochs": 1,
            "batch_size": 8,
            "seed": 42,
            "device": "cpu", # Use CPU for testing
            "metadata_path": "data/processed/metadata.json",
            "model_dir": str(tmp_path), # Use temporary directory for model output
            "wandb": {"enabled": False, "entity": "test", "project": "test"},
            "artifact": {"name": "test_model", "type": "model", "description": "test model"},
            "target": "class", # Assuming 'class' as the default target for the dummy data
        },
        "model": {
            "in_channels": 1, # Dummy data is 1 channel
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
            "data_dir": "data", # Path to dummy data
        }
    })
    # Dynamically set num_classes from metadata, similar to how train does it
    with open("data/processed/metadata.json") as f:
        metadata = json.load(f)
    cfg.experiments.num_classes = len(metadata[f"{cfg.experiments.target}_to_idx"])

    # Ensure a clean slate for model output
    model_path = tmp_path / "model.pth"
    if model_path.exists():
        os.remove(model_path)

    # First run
    _train_model(cfg) # Calls the patched train function
    # num_classes from dummy data in test_data.py (5 for "class")
    state_dict_1 = get_model_state_dict(model_path, num_classes=5)
    os.remove(model_path)

    # Second run
    _train_model(cfg) # Calls the patched train function again
    # num_classes from dummy data in test_data.py (5 for "class")
    state_dict_2 = get_model_state_dict(model_path, num_classes=5)

    # Check that the model weights are the same
    for key in state_dict_1:
        assert torch.all(torch.eq(state_dict_1[key], state_dict_2[key]))
