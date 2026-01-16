import torch

from src.plants.model import Model


def test_model_initialization():
    """Test that the model can be initialized with a specific number of classes."""
    model = Model(
        num_classes=10,
        in_channels=3,
        conv1_out=32,
        conv1_kernel=3,
        conv1_stride=1,
        conv2_out=64,
        conv2_kernel=3,
        conv2_stride=1,
        conv2_padding=1,
        dropout=0.5,
    )
    assert model.fc1.out_features == 10, "The number of output classes should be 10"


def test_model_forward_pass():
    """Test the output shape of the model's forward pass."""
    model = Model(
        num_classes=10,
        in_channels=3,
        conv1_out=32,
        conv1_kernel=3,
        conv1_stride=1,
        conv2_out=64,
        conv2_kernel=3,
        conv2_stride=1,
        conv2_padding=1,
        dropout=0.5,
    )
    # Create a dummy input tensor with the expected shape (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 10), "The output shape should be (batch_size, num_classes)"
