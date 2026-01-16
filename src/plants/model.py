import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn


class Model(nn.Module):
    """My awesome model. currently setup for plant target classification."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
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
        self.conv1 = nn.Conv2d(in_channels, conv1_out, conv1_kernel, conv1_stride)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, conv2_kernel, conv2_stride, conv2_padding)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(conv2_out, num_classes)
        self.dropout = nn.Dropout2d(dropout)

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


@hydra.main(version_base=None, config_path="../../configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    print(f"model configuration:\n{OmegaConf.to_yaml(cfg.model)}")
    if cfg.model.num_classes is None:
        raise ValueError("cfg.model.num_classes is None. Set experiment.num_classes or run train.py to derive it.")
    model = Model(
        num_classes=cfg.model.num_classes,
        in_channels=cfg.model.in_channels,
        conv1_out=cfg.model.conv1_out,
        conv1_kernel=cfg.model.conv1_kernel,
        conv1_stride=cfg.model.conv1_stride,
        conv2_out=cfg.model.conv2_out,
        conv2_kernel=cfg.model.conv2_kernel,
        conv2_stride=cfg.model.conv2_stride,
        conv2_padding=cfg.model.conv2_padding,
        dropout=cfg.model.dropout,
    )
    x = torch.rand(1, cfg.model.in_channels, 224, 224)
    print(model(x).shape)


if __name__ == "__main__":
    main()
