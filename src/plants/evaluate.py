import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

import wandb

METADATA_PATH = Path("data/processed/metadata.json")

def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def _load_config(config_name: str = "default_config") -> DictConfig:
    config_dir = _repo_root() / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name)


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


def _class_metadata(metadata: dict, target: str) -> tuple[int, list[str]]:
    if target == "class":
        mapping = metadata["class_to_idx"]
    elif target == "disease":
        mapping = metadata["disease_to_idx"]
    elif target == "plant":
        mapping = metadata["plant_to_idx"]
    else:
        raise ValueError(
            f"Unsupported target '{target}'. Expected one of ['class', 'disease', 'plant'] for evaluation."
        )

    class_names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    return len(mapping), class_names


def evaluate(
    model_checkpoint: Annotated[
        Optional[Path],
        typer.Argument(help="Path to model checkpoint (defaults to config experiments.model_dir/model.pth)."),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", "-b", help="Override batch size from config."),
    ] = None,
    target: Annotated[
        Optional[str],
        typer.Option("--target", help="Override target from config."),
    ] = None,
    data_dir: Annotated[
        Optional[Path],
        typer.Option("--data-dir", help="Override data directory from config."),
    ] = None,
    config_name: Annotated[str, typer.Option("--config-name", help="Hydra config name to load.")] = "default_config",
) -> None:
    """Evaluate a trained model."""
    from src.plants.data import MyDataset
    from src.plants.model import Model

    cfg = _load_config(config_name)
    hparams = cfg.experiments

    if batch_size is not None:
        hparams.batch_size = batch_size
    if target is not None:
        hparams.target = target
    if data_dir is not None:
        cfg.dataloader.data_dir = str(data_dir)

    device = _select_device(hparams.device)
    base_dir = _repo_root()
    resolved_data_dir = _resolve_path(cfg.dataloader.data_dir, base_dir)
    metadata_path = _resolve_path(hparams.metadata_path, base_dir)
    model_dir = _resolve_path(hparams.model_dir, base_dir)
    checkpoint_path = _resolve_path(model_checkpoint or model_dir / "model.pth", base_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    with open(metadata_path) as f:
        metadata = json.load(f)
    num_classes, class_names = _class_metadata(metadata, hparams.target)

    state = torch.load(checkpoint_path, map_location="cpu")
    state_num_classes = int(state["fc.weight"].shape[0])
    state_in_channels = int(state["layer1.0.weight"].shape[1])
    if state_num_classes != num_classes:
        print(
            f"Warning: checkpoint has {state_num_classes} classes, but metadata has {num_classes}. "
            "Using checkpoint classes for evaluation."
        )
        num_classes = state_num_classes
        class_names = [f"class_{idx}" for idx in range(num_classes)]
    if state_in_channels != int(cfg.model.in_channels):
        print(
            f"Warning: checkpoint expects {state_in_channels} input channel(s), but config has "
            f"{cfg.model.in_channels}. Using checkpoint channels for evaluation."
        )

    model = Model(
        num_classes=num_classes,
        in_channels=state_in_channels,
        conv1_out=cfg.model.conv1_out,
        conv1_kernel=cfg.model.conv1_kernel,
        conv1_stride=cfg.model.conv1_stride,
        conv2_out=cfg.model.conv2_out,
        conv2_kernel=cfg.model.conv2_kernel,
        conv2_stride=cfg.model.conv2_stride,
        conv2_padding=cfg.model.conv2_padding,
        dropout=cfg.model.dropout,
    )
    model.load_state_dict(state)
    model.to(device)

    dataset = MyDataset(resolved_data_dir, target=hparams.target)
    _, val_set = dataset.load_plantvillage(target=hparams.target)
    val_dataloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
    )

    run = None
    if hparams.wandb.enabled:
        run = wandb.init(
            entity=hparams.wandb.entity,
            project=hparams.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            job_type="evaluate",
        )

    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total_loss = 0.0
    total_count = 0

    with torch.inference_mode():
        for img, target_tensor in val_dataloader:
            img = img.to(device)
            target_tensor = target_tensor.to(device)
            y_pred = model(img)
            loss = loss_fn(y_pred, target_tensor)
            total_loss += loss.item() * target_tensor.size(0)
            total_count += target_tensor.size(0)
            all_preds.append(y_pred.detach().cpu())
            all_targets.append(target_tensor.detach().cpu())

    preds_tensor = torch.cat(all_preds, 0)
    targets_tensor = torch.cat(all_targets, 0)
    preds = preds_tensor.argmax(dim=1).cpu().numpy()
    targets = targets_tensor.cpu().numpy()

    avg_loss = total_loss / max(total_count, 1)
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average="weighted", zero_division=0)
    recall = recall_score(targets, preds, average="weighted", zero_division=0)
    f1 = f1_score(targets, preds, average="weighted", zero_division=0)

    print(f"Eval loss: {avg_loss:.4f}")
    print(f"Eval accuracy: {accuracy:.4f}")

    if run is not None:
        run.summary.update(
            {
                "eval_loss": avg_loss,
                "eval_accuracy": accuracy,
                "eval_precision": precision,
                "eval_recall": recall,
                "eval_f1": f1,
            }
        )
        wandb.log(
            {
                "eval_loss": avg_loss,
                "eval_accuracy": accuracy,
                "eval_precision": precision,
                "eval_recall": recall,
                "eval_f1": f1,
            }
        )

        wandb.log(
            {
                "eval_confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=targets.tolist(), preds=preds.tolist(), class_names=class_names
                )
            }
        )

        report = classification_report(
            targets,
            preds,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        table = wandb.Table(columns=["label", "precision", "recall", "f1", "support"])
        for label, metrics in report.items():
            if label in {"accuracy", "macro avg", "weighted avg"}:
                continue
            table.add_data(
                label,
                metrics["precision"],
                metrics["recall"],
                metrics["f1-score"],
                metrics["support"],
            )
        wandb.log({"eval_class_report": table})

    print("Evaluation complete")


if __name__ == "__main__":
    _ensure_repo_root_on_path()
    typer.run(evaluate)
