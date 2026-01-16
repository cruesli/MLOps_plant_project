import json
from pathlib import Path

import hydra
import torch
from hydra.utils import to_absolute_path
from model import Model
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import wandb
from data import MyDataset


def _select_device(preference: str) -> torch.device:
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_num_classes(target: str, metadata_path: Path) -> int:
    with open(metadata_path) as handle:
        metadata = json.load(handle)
    if target == "class":
        return len(metadata["class_to_idx"])
    if target == "disease":
        return len(metadata["disease_to_idx"])
    if target == "plant":
        return len(metadata["plant_to_idx"])
    raise ValueError(f"Unsupported target '{target}'. Expected one of ['class', 'disease', 'plant'] for evaluation.")


@hydra.main(version_base=None, config_path="../../configs", config_name="default_config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model."""
    print(f"configuration: \n{OmegaConf.to_yaml(cfg)}")
    hparams = cfg.experiments
    device = _select_device(hparams.device)

    data_dir = to_absolute_path(cfg.dataloader.data_dir)
    metadata_path = Path(to_absolute_path(hparams.metadata_path))
    model_dir = Path(to_absolute_path(hparams.model_dir))
    model_checkpoint = getattr(hparams, "model_checkpoint", None)
    checkpoint_path = Path(to_absolute_path(model_checkpoint)) if model_checkpoint else model_dir / "model.pth"
    if not checkpoint_path.exists():
        msg = f"Model checkpoint not found at {checkpoint_path}. Train a model or set experiments.model_checkpoint."
        raise FileNotFoundError(msg)

    target = hparams.target
    num_classes = _resolve_num_classes(target, metadata_path)

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    run = None
    if hparams.wandb.enabled:
        run = wandb.init(
            entity=hparams.wandb.entity,
            project=hparams.wandb.project,
            config=wandb_config,
        )

    model = Model(
        num_classes=num_classes,
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
    model.to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)

    dataset = MyDataset(data_dir, target=target)
    _, val_set = dataset.load_plantvillage(target=target)
    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
    )

    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for img, target_batch in val_dataloader:
            img, target_batch = img.to(device), target_batch.to(device)
            y_pred = model(img)
            preds.append(y_pred.detach().cpu())
            targets.append(target_batch.detach().cpu())

    preds_tensor = torch.cat(preds, 0)
    targets_tensor = torch.cat(targets, 0)

    val_accuracy = accuracy_score(targets_tensor, preds_tensor.argmax(dim=1))
    val_precision = precision_score(targets_tensor, preds_tensor.argmax(dim=1), average="weighted")
    val_recall = recall_score(targets_tensor, preds_tensor.argmax(dim=1), average="weighted")
    val_f1 = f1_score(targets_tensor, preds_tensor.argmax(dim=1), average="weighted")

    print(f"Validation accuracy: {val_accuracy}")
    print(f"Validation precision: {val_precision}")
    print(f"Validation recall: {val_recall}")
    print(f"Validation f1: {val_f1}")
    if run is not None:
        wandb.log(
            {
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            }
        )
        run.summary["val_accuracy"] = val_accuracy
        run.summary["val_precision"] = val_precision
        run.summary["val_recall"] = val_recall
        run.summary["val_f1"] = val_f1
    print("Evaluation complete")


if __name__ == "__main__":
    evaluate()
