import torch
from plants.train import train
from plants.model import Model
import os

def get_model_state_dict(model_path):
    """Load and return the state dict of a model."""
    # First, need to initialize a model with the correct architecture
    
    # Read metadata to get num_classes
    import json
    from pathlib import Path
    
    with open("data/processed/metadata.json") as f:
        metadata = json.load(f)
    num_classes = len(metadata["class_to_idx"])
    
    model = Model(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    return model.state_dict()

def test_reproducibility():
    """Test the reproducibility of the training script."""
    # Train the model twice with the same seed
    train(epochs=1, batch_size=8, lr=1e-3, seed=42)
    state_dict_1 = get_model_state_dict("models/model.pth")
    os.remove("models/model.pth") # remove the model to ensure the next run starts from scratch

    train(epochs=1, batch_size=8, lr=1e-3, seed=42)
    state_dict_2 = get_model_state_dict("models/model.pth")

    # Check that the model weights are the same
    for key in state_dict_1:
        assert torch.all(torch.eq(state_dict_1[key], state_dict_2[key]))
