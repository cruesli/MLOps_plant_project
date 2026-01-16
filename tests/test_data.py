import torch
from torch.utils.data import Dataset, TensorDataset
import json
from pathlib import Path
import pytest

from src.plants.data import MyDataset


def test_my_dataset():
    """Test the MyDataset class."""
    # To run this test, we need to have the data preprocessed.
    # We will create a dummy preprocessed dataset.

    # Create dummy data and metadata
    dummy_train_images = torch.randn(10, 1, 28, 28)
    dummy_train_labels = torch.randint(0, 5, (10,))
    dummy_train_disease = torch.randint(0, 2, (10,))
    dummy_train_plant = torch.randint(0, 3, (10,))

    dummy_val_images = torch.randn(5, 1, 28, 28)
    dummy_val_labels = torch.randint(0, 5, (5,))
    dummy_val_disease = torch.randint(0, 2, (5,))
    dummy_val_plant = torch.randint(0, 3, (5,))

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

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

    dataset = MyDataset("data", split="train", target="class")
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 10

    train_ds, val_ds = dataset.load_plantvillage(target="class")
    assert isinstance(train_ds, TensorDataset)
    assert isinstance(val_ds, TensorDataset)
    assert len(train_ds) == 10
    assert len(val_ds) == 5


def test_dataset_getitem():
    """Test the __getitem__ method of the MyDataset class."""
    dataset = MyDataset("data", split="train", target="class")
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert img.shape == (1, 28, 28)
    assert label.shape == ()


def test_data_structure_and_amount():
    """Test the structure and amount of the data."""
    train_images = torch.load("data/processed/train_images.pt")
    train_labels = torch.load("data/processed/train_labels.pt")
    train_disease = torch.load("data/processed/train_disease_labels.pt")
    train_plant = torch.load("data/processed/train_plant_labels.pt")

    val_images = torch.load("data/processed/val_images.pt")
    val_labels = torch.load("data/processed/val_labels.pt")
    val_disease = torch.load("data/processed/val_disease_labels.pt")
    val_plant = torch.load("data/processed/val_plant_labels.pt")

    assert train_images.shape == (10, 1, 28, 28)
    assert train_labels.shape == (10,)
    assert train_disease.shape == (10,)
    assert train_plant.shape == (10,)

    assert val_images.shape == (5, 1, 28, 28)
    assert val_labels.shape == (5,)
    assert val_disease.shape == (5,)
    assert val_plant.shape == (5,)

    with open("data/processed/metadata.json") as f:
        metadata = json.load(f)
    
    assert len(metadata["class_to_idx"]) == 5
    assert len(metadata["disease_to_idx"]) == 2
    assert len(metadata["plant_to_idx"]) == 3


def test_dataset_initialization():
    """Test the initialization of the MyDataset class."""
    # Test with different split and target values
    MyDataset("data", split="train", target="class")
    MyDataset("data", split="val", target="disease")
    MyDataset("data", split="train", target="plant")
    MyDataset("data", split="val", target="both")
    MyDataset("data", split="train", target="all")

    # Test with invalid target
    with pytest.raises(ValueError):
        MyDataset("data", split="train", target="invalid_target")