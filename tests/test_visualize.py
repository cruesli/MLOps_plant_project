from pathlib import Path

import torch
from omegaconf import OmegaConf

import src.plants.model as model_module
from src.plants import visualize as visualize_module


class DummyEmbeddingModel(torch.nn.Module):
    """Small model that matches visualize() checkpoint expectations."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        conv1_out: int = 8,
        conv1_kernel: int = 3,
        conv1_stride: int = 1,
        conv2_out: int = 16,  # Unused, kept for signature parity
        conv2_kernel: int = 3,  # Unused, kept for signature parity
        conv2_stride: int = 1,  # Unused, kept for signature parity
        conv2_padding: int = 1,
        dropout: float = 0.0,  # Unused, kept for signature parity
    ) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels, conv1_out, kernel_size=conv1_kernel, stride=conv1_stride, padding=conv2_padding
        )
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(conv1_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)


def test_visualize_creates_embedding_plot(tmp_path, dummy_processed_data, monkeypatch):
    """Run visualize() end-to-end on fixture data and ensure output is saved."""
    processed_dir = Path(dummy_processed_data) / "processed"

    # Expand validation split to satisfy t-SNE perplexity requirements.
    val_size = 40
    torch.manual_seed(0)
    torch.save(torch.randn(val_size, 3, 28, 28), processed_dir / "val_images.pt")
    torch.save(torch.randint(0, 5, (val_size,)), processed_dir / "val_labels.pt")
    torch.save(torch.randint(0, 2, (val_size,)), processed_dir / "val_disease_labels.pt")
    torch.save(torch.randint(0, 3, (val_size,)), processed_dir / "val_plant_labels.pt")

    num_classes = 5
    model_params = {
        "in_channels": 3,
        "conv1_out": 8,
        "conv1_kernel": 3,
        "conv1_stride": 1,
        "conv2_out": 16,
        "conv2_kernel": 3,
        "conv2_stride": 1,
        "conv2_padding": 1,
        "dropout": 0.0,
    }
    checkpoint_path = tmp_path / "dummy_embeddings.pth"
    dummy_model = DummyEmbeddingModel(num_classes=num_classes, **model_params)
    torch.save(dummy_model.state_dict(), checkpoint_path)

    cfg = OmegaConf.create(
        {
            "experiments": {
                "device": "cpu",
                "target": "class",
                "metadata_path": str(processed_dir / "metadata.json"),
                "model_dir": str(tmp_path),
                "wandb": {"enabled": False, "entity": "local", "project": "local"},
            },
            "dataloader": {"data_dir": str(dummy_processed_data), "shuffle": False, "num_workers": 0},
            "model": {"num_classes": num_classes, **model_params},
        }
    )

    monkeypatch.setattr(model_module, "Model", DummyEmbeddingModel)
    monkeypatch.setattr(visualize_module, "_load_config", lambda config_name="default_config": cfg)
    monkeypatch.setattr(visualize_module, "_repo_root", lambda: tmp_path)

    figure_name = "test_embeddings.png"
    visualize_module.visualize(
        model_checkpoint=checkpoint_path,
        figure_name=figure_name,
        target=cfg.experiments.target,
        data_dir=cfg.dataloader.data_dir,
        config_name="test_config",
    )

    figure_path = tmp_path / "reports" / "figures" / figure_name
    assert figure_path.exists()
