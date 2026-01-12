import torch
from torch import nn

class Model(nn.Module):
    """My awesome model. currently setup for plant target classification."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 3, 1)
        self.fc1 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)
    
if __name__ == "__main__":
    model = Model()
    x = torch.rand(1, 1, 28, 28)
    print(f"Model architecture: {model}")
    print(f"n params: {sum(p.numel() for p in model.parameters())}")
    print(f"Output shape of model: {model(x).shape}")
