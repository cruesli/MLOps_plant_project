import json
from pathlib import Path
import pytest
import torch
from torch.utils.data import Dataset, TensorDataset
from src.plants.data import MyDataset

def test_my_dataset(dummy_processed_data):
    """Test the MyDataset class."""
    dataset = MyDataset(dummy_processed_data, split="train", target="class")
    assert isinstance(dataset, Dataset)
    assert len(dataset) == 10

    train_ds, val_ds = dataset.load_plantvillage(target="class")
    assert isinstance(train_ds, TensorDataset)
    assert isinstance(val_ds, TensorDataset)
    assert len(train_ds) == 10
    assert len(val_ds) == 5

def test_dataset_getitem(dummy_processed_data):
    """Test the __getitem__ method of the MyDataset class."""
    dataset = MyDataset(dummy_processed_data, split="train", target="class")
    img, label = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert img.shape == (3, 28, 28)
    assert label.shape == ()

def test_data_structure_and_amount(dummy_processed_data):
    """Test the structure and amount of the data."""
    processed_dir = dummy_processed_data / "processed"
    train_images = torch.load(processed_dir / "train_images.pt")
    train_labels = torch.load(processed_dir / "train_labels.pt")
    train_disease = torch.load(processed_dir / "train_disease_labels.pt")
    train_plant = torch.load(processed_dir / "train_plant_labels.pt")

    val_images = torch.load(processed_dir / "val_images.pt")
    val_labels = torch.load(processed_dir / "val_labels.pt")
    val_disease = torch.load(processed_dir / "val_disease_labels.pt")
    val_plant = torch.load(processed_dir / "val_plant_labels.pt")

    assert train_images.shape == (10, 3, 28, 28)
    assert train_labels.shape == (10,)
    assert train_disease.shape == (10,)
    assert train_plant.shape == (10,)

    assert val_images.shape == (5, 3, 28, 28)
    assert val_labels.shape == (5,)
    assert val_disease.shape == (5,)
    assert val_plant.shape == (5,)

    with open(processed_dir / "metadata.json") as f:
        metadata = json.load(f)

    assert len(metadata["class_to_idx"]) == 5
    assert len(metadata["disease_to_idx"]) == 2
    assert len(metadata["plant_to_idx"]) == 3

def test_dataset_initialization(dummy_processed_data):
    """Test the initialization of the MyDataset class."""
    # Test with different split and target values
    MyDataset(dummy_processed_data, split="train", target="class")
    MyDataset(dummy_processed_data, split="val", target="disease")
    MyDataset(dummy_processed_data, split="train", target="plant")
    MyDataset(dummy_processed_data, split="val", target="both")
    MyDataset(dummy_processed_data, split="train", target="all")

    # Test with invalid target
    with pytest.raises(ValueError):
        MyDataset(dummy_processed_data, split="train", target="invalid_target")

def test_missing_data_file(dummy_processed_data):
    """Test that MyDataset raises an error if a data file is missing."""
    processed_dir = dummy_processed_data / "processed"
    (processed_dir / "train_labels.pt").unlink()

    with pytest.raises(FileNotFoundError):
        MyDataset(dummy_processed_data, split="train", target="class")

def test_corrupted_data_file(dummy_processed_data):
    """Test that MyDataset raises an error if a data file is corrupted."""
    processed_dir = dummy_processed_data / "processed"
    with open(processed_dir / "train_labels.pt", "w") as f:
        f.write("this is not a tensor")

    with pytest.raises(Exception): # torch.load can raise different errors
        MyDataset(dummy_processed_data, split="train", target="class")
