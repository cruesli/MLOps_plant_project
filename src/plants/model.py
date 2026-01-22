import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn


class Model(nn.Module):
    """Refactored model with Batch Normalization and organized blocks."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        conv1_out: int = 32,
        conv1_kernel: int = 3,
        conv1_stride: int = 1,
        conv2_out: int = 64,
        conv2_kernel: int = 3,
        conv2_stride: int = 1,
        conv2_padding: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        # Block 1: Initial Feature Extraction
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, conv1_out, conv1_kernel, stride=conv1_stride, padding=1),
            nn.BatchNorm2d(conv1_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: Deeper Features
        self.layer2 = nn.Sequential(
            nn.Conv2d(conv1_out, conv2_out, conv2_kernel, stride=conv2_stride, padding=conv2_padding),
            nn.BatchNorm2d(conv2_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(conv2_out, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


@hydra.main(version_base=None, config_path="../../configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    # Use OmegaConf.to_container for easier dictionary-style unpacking
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    if model_params.get("num_classes") is None:
        raise ValueError("cfg.model.num_classes is missing.")

    # Optimized instantiation: Unpacking dictionary directly
    model = Model(**model_params)
    # Efficiency check: Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x = torch.rand(1, cfg.model.in_channels, 224, 224).to(device)
    print(f"Output shape: {model(x).shape}")


if __name__ == "__main__":
    main()
