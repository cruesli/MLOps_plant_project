import json
import pytest
import torch

@pytest.fixture
def dummy_processed_data(tmp_path):
    """Create a dummy processed dataset and return the path to it."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy data and metadata
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
    return tmp_path
