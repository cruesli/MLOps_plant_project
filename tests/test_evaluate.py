import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.plants import evaluate as evaluate_module
from src.plants.model import Model


def test_evaluate_runs_with_dummy_checkpoint(tmp_path, dummy_processed_data, monkeypatch, capsys):
    """Verify evaluate() completes on fixture data with a lightweight checkpoint."""
    processed_dir = Path(dummy_processed_data) / "processed"
    metadata = json.loads((processed_dir / "metadata.json").read_text())
    num_classes = len(metadata["class_to_idx"])

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

    checkpoint_path = tmp_path / "model.pth"
    torch.manual_seed(0)
    dummy_model = Model(num_classes=num_classes, **model_params)
    torch.save(dummy_model.state_dict(), checkpoint_path)

    cfg = OmegaConf.create(
        {
            "experiments": {
                "batch_size": 4,
                "device": "cpu",
                "target": "class",
                "metadata_path": str(processed_dir / "metadata.json"),
                "model_dir": str(tmp_path),
                "wandb": {"enabled": False, "entity": "local", "project": "local"},
                "artifact": {"name": "dummy", "type": "model", "description": "local test checkpoint"},
            },
            "dataloader": {"data_dir": str(dummy_processed_data), "shuffle": False, "num_workers": 0},
            "model": {"num_classes": num_classes, **model_params},
        }
    )

    monkeypatch.setattr(evaluate_module, "_load_config", lambda config_name="default_config": cfg)
    monkeypatch.setattr(evaluate_module, "_repo_root", lambda: tmp_path)

    evaluate_module.evaluate(
        model_checkpoint=checkpoint_path,
        batch_size=cfg.experiments.batch_size,
        target=cfg.experiments.target,
        data_dir=cfg.dataloader.data_dir,
        config_name="test_config",
    )

    captured = capsys.readouterr()
    assert "Evaluation complete" in captured.out
