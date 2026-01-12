import torch

from plants.model import Model


def test_model_initialization():
    """Test that the model can be initialized with a specific number of classes."""
    model = Model(num_classes=10)
    assert model.fc1.out_features == 10, "The number of output classes should be 10"


def test_model_forward_pass():
    """Test the output shape of the model's forward pass."""
    model = Model(num_classes=10)
    # Create a dummy input tensor with the expected shape (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    assert output.shape == (1, 10), "The output shape should be (batch_size, num_classes)"
