import torch
from torch import nn

class Model(nn.Module):
    """My awesome model. currently setup for plant target classification."""
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 3, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = self.dropout(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)

    
if __name__ == "__main__":
    model = Model(num_classes=14)  # or 21/38 depending on target
    x = torch.rand(1, 3, 224, 224)
    print(model(x).shape)

