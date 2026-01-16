import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score

import wandb
from src.plants.data import MyDataset
from src.plants.model import Model


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


@hydra.main(version_base=None, config_path="../../configs", config_name="default_config")
def train(cfg: DictConfig) -> None:
    print(f"configuration: \n{OmegaConf.to_yaml(cfg)}")
    hparams = cfg.experiments
    torch.manual_seed(hparams.seed)
    device = _select_device(hparams.device)

    data_dir = to_absolute_path(cfg.dataloader.data_dir)
    metadata_path = to_absolute_path(hparams.metadata_path)
    model_dir = Path(to_absolute_path(hparams.model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)

    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    run = None
    if hparams.wandb.enabled:
        run = wandb.init(
            entity=hparams.wandb.entity,
            project=hparams.wandb.project,
            config=wandb_config,
        )

    target = hparams.target
    dataset = MyDataset(data_dir, target=target)

    # Read metadata to get num_classes
    with open(metadata_path) as f:
        metadata = json.load(f)
    if target == "class":
        num_classes = len(metadata["class_to_idx"])
    elif target == "disease":
        num_classes = len(metadata["disease_to_idx"])
    elif target == "plant":
        num_classes = len(metadata["plant_to_idx"])
    else:
        raise ValueError(f"Unsupported target '{target}'. Expected one of ['class', 'disease', 'plant'] for training.")
    hparams.num_classes = num_classes

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
    train_set, _ = dataset.load_plantvillage(target=target)
    train_dataloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=hparams.batch_size,
        shuffle=cfg.dataloader.shuffle,
        num_workers=cfg.dataloader.num_workers,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    statistics = {"train_loss": [], "train_accuracy": []}
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for epoch in range(hparams.epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            if run is not None:
                wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}, accuracy: {accuracy}")

                if run is not None:
                    images = [
                        wandb.Image(
                            (single_img.detach().cpu().clamp(0, 1) * 255).to(torch.uint8),
                            caption=f"Input {idx}",
                        )
                        for idx, single_img in enumerate(img[:5])
                    ]
                    wandb.log({"input_images": images})

                    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                    wandb.log({"gradients": wandb.Histogram(grads.detach().cpu().numpy())})

                statistics["train_loss"].append(loss.item())
                statistics["train_accuracy"].append(accuracy)
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
        wandb.log({"roc": wandb.Image(fig_roc)})
    plt.close(fig_roc)

    final_accuracy = accuracy_score(targets_tensor.cpu(), preds_tensor.argmax(dim=1).cpu())
    final_precision = precision_score(targets_tensor.cpu(), preds_tensor.argmax(dim=1).cpu(), average="weighted")
    final_recall = recall_score(targets_tensor.cpu(), preds_tensor.argmax(dim=1).cpu(), average="weighted")
    final_f1 = f1_score(targets_tensor.cpu(), preds_tensor.argmax(dim=1).cpu(), average="weighted")

    model_path = model_dir / "model.pth"
    torch.save(model.state_dict(), model_path)
    if run is not None:
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


if __name__ == "__main__":
    train()
