import json
import os
import sys
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score

import wandb


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


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


def _normalize_gcs_prefix(prefix: str) -> str:
    return prefix[5:] if prefix.startswith("gs://") else prefix


def _maybe_upload_model(model_path: Path, run_id: str) -> None:
    gcs_prefix = os.environ.get("PLANTS_MODEL_GCS")
    if not gcs_prefix:
        return

    try:
        from gcsfs import GCSFileSystem
    except ImportError as exc:  # pragma: no cover - only needed in cloud jobs
        raise RuntimeError("gcsfs is required to upload models to GCS.") from exc

    fs = GCSFileSystem()
    remote_base = _normalize_gcs_prefix(gcs_prefix.rstrip("/"))
    remote_path = f"{remote_base}/{run_id}/{model_path.name}"
    fs.put(str(model_path), remote_path)


def _maybe_download_processed_data(data_dir: Path, metadata_path: Path) -> None:
    if metadata_path.exists():
        return

    gcs_prefix = os.environ.get("PLANTS_DATA_GCS")
    if not gcs_prefix:
        return

    try:
        from gcsfs import GCSFileSystem
    except ImportError as exc:  # pragma: no cover - only needed in cloud jobs
        raise RuntimeError("gcsfs is required to download data from GCS.") from exc

    fs = GCSFileSystem()
    remote_base = _normalize_gcs_prefix(gcs_prefix.rstrip("/"))
    if fs.exists(f"{remote_base}/metadata.json"):
        remote_prefix = remote_base
    else:
        remote_prefix = f"{remote_base}/processed"

    local_processed = data_dir / "processed"
    local_processed.mkdir(parents=True, exist_ok=True)

    for remote_path in fs.ls(remote_prefix):
        if remote_path.endswith("/"):
            continue
        filename = Path(remote_path).name
        if not (filename.endswith(".pt") or filename.endswith(".json")):
            continue
        fs.get(remote_path, str(local_processed / filename))

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata at {metadata_path}. Set PLANTS_DATA_GCS to a GCS path "
            "containing processed tensors and metadata.json."
        )


def _train_model(cfg: DictConfig) -> None:  # Renamed and now can be tested directly
    from src.plants.data import MyDataset
    from src.plants.model import Model

    print(f"configuration: \n{OmegaConf.to_yaml(cfg)}")
    hparams = cfg.experiments
    torch.manual_seed(hparams.seed)  # My change: set the seed
    device = _select_device(hparams.device)

    data_dir = to_absolute_path(cfg.dataloader.data_dir)
    metadata_path = to_absolute_path(hparams.metadata_path)
    model_dir = Path(to_absolute_path(hparams.model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)

    _maybe_download_processed_data(Path(data_dir), Path(metadata_path))

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    run = None
    if hparams.wandb.enabled:
        run = wandb.init(
            entity=hparams.wandb.entity,
            project=hparams.wandb.project,
            config=wandb_config,
            job_type="train",
        )
        wandb.define_metric("*", step_metric="global_step")
        sweep_lr = run.config.get("lr")
        sweep_batch = run.config.get("batch_size")
        sweep_dropout = run.config.get("dropout")
        if sweep_lr is not None:
            hparams.lr = float(sweep_lr)
        if sweep_batch is not None:
            hparams.batch_size = int(sweep_batch)
        if sweep_dropout is not None:
            cfg.model.dropout = float(sweep_dropout)

    target = hparams.target
    dataset = MyDataset(data_dir, target=target)

    # Read metadata to get num_classes
    with open(metadata_path) as f:
        metadata = json.load(f)
    if target == "class":
        num_classes = len(metadata["class_to_idx"])
        class_names = [name for name, _ in sorted(metadata["class_to_idx"].items(), key=lambda item: item[1])]
    elif target == "disease":
        num_classes = len(metadata["disease_to_idx"])
        class_names = [name for name, _ in sorted(metadata["disease_to_idx"].items(), key=lambda item: item[1])]
    elif target == "plant":
        num_classes = len(metadata["plant_to_idx"])
        class_names = [name for name, _ in sorted(metadata["plant_to_idx"].items(), key=lambda item: item[1])]
    else:
        raise ValueError(f"Unsupported target '{target}'. Expected one of ['class', 'disease', 'plant'] for training.")
    hparams.num_classes = num_classes
    mean_val = metadata.get("mean")
    std_val = metadata.get("std")
    mean_tensor = torch.tensor(float(mean_val)) if mean_val is not None else None
    std_tensor = torch.tensor(float(std_val)) if std_val is not None else None

    train_set, _ = dataset.load_plantvillage(target=target)
    data_channels = int(train_set.tensors[0].shape[1])
    in_channels = int(cfg.model.in_channels)
    if data_channels != in_channels:
        print(
            f"Warning: data has {data_channels} channel(s), but config expects {in_channels}. "
            f"Using {data_channels} for this run."
        )
        in_channels = data_channels

    model = Model(
        num_classes=num_classes,
        in_channels=in_channels,
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
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=hparams.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    statistics = {"train_loss": [], "train_accuracy": []}
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    global_step = 0

    for epoch in range(hparams.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            running_loss += loss.item() * target.size(0)
            running_correct += (y_pred.argmax(dim=1) == target).float().sum().item()
            running_total += target.size(0)
            if run is not None:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_accuracy": accuracy,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "global_step": global_step,
                    },
                )
            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}, accuracy: {accuracy}")

                if run is not None:
                    images = []
                    for idx, single_img in enumerate(img[:5]):
                        image_cpu = single_img.detach().cpu()
                        if mean_tensor is not None and std_tensor is not None:
                            image_cpu = image_cpu * std_tensor + mean_tensor
                        image_cpu = image_cpu.clamp(0, 1)
                        images.append(wandb.Image((image_cpu * 255).to(torch.uint8), caption=f"Input {idx}"))
                    wandb.log({"input_images": images, "global_step": global_step})

                    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                    wandb.log(
                        {
                            "gradients": wandb.Histogram(grads.detach().cpu().numpy()),
                            "global_step": global_step,
                        }
                    )

                statistics["train_loss"].append(loss.item())
                statistics["train_accuracy"].append(accuracy)
            global_step += 1
        if run is not None and running_total:
            epoch_loss = running_loss / running_total
            epoch_accuracy = running_correct / running_total
            wandb.log(
                {
                    "epoch": epoch,
                    "train_epoch_loss": epoch_loss,
                    "train_epoch_accuracy": epoch_accuracy,
                    "global_step": global_step,
                },
            )
            scheduler.step()
    print("Training complete")

    # Concatenate stored predictions/targets
    preds_tensor = torch.cat(preds, 0)
    targets_tensor = torch.cat(targets, 0)

    # ROC curves per class
    fig_roc, ax_roc = plt.subplots()
    for class_id in range(num_classes):
        one_hot = torch.zeros_like(targets_tensor)
        one_hot[targets_tensor == class_id] = 1
        RocCurveDisplay.from_predictions(
            one_hot,
            preds_tensor[:, class_id],
            name=f"ROC curve for {class_id}",
            plot_chance_level=(class_id == 2),
            ax=ax_roc,
        )
    if run is not None:
        if class_names:
            wandb.log(
                {
                    "train_confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=targets_tensor.cpu().tolist(),
                        preds=preds_tensor.argmax(dim=1).cpu().tolist(),
                        class_names=class_names,
                    ),
                    "global_step": global_step,
                }
            )
        wandb.log({"roc": wandb.Image(fig_roc), "global_step": global_step})
    plt.close(fig_roc)

    final_accuracy = accuracy_score(targets_tensor.cpu(), preds_tensor.argmax(dim=1).cpu())
    final_precision = precision_score(targets_tensor.cpu(), preds_tensor.argmax(dim=1).cpu(), average="weighted")
    final_recall = recall_score(targets_tensor.cpu(), preds_tensor.argmax(dim=1).cpu(), average="weighted")
    final_f1 = f1_score(targets_tensor.cpu(), preds_tensor.argmax(dim=1).cpu(), average="weighted")

    model_path = model_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    if run is not None:
        run.summary.update(
            {
                "final_accuracy": final_accuracy,
                "final_precision": final_precision,
                "final_recall": final_recall,
                "final_f1": final_f1,
            }
        )
        wandb.log(
            {
                "final_accuracy": final_accuracy,
                "final_precision": final_precision,
                "final_recall": final_recall,
                "final_f1": final_f1,
                "global_step": global_step,
            }
        )
        artifact = wandb.Artifact(
            name=hparams.artifact.name,
            type=hparams.artifact.type,
            description=hparams.artifact.description,
            metadata={
                "accuracy": final_accuracy,
                "precision": final_precision,
                "recall": final_recall,
                "f1": final_f1,
            },
        )
        artifact.add_file(str(model_path))
        run.log_artifact(artifact)
        _maybe_upload_model(model_path, run.id)
    else:
        _maybe_upload_model(model_path, "local")


@hydra.main(version_base=None, config_path="../../configs", config_name="default_config")
def train(cfg: DictConfig) -> None:
    _train_model(cfg)


if __name__ == "__main__":
    _ensure_repo_root_on_path()
    train()
