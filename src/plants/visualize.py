import json
import sys
from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import torch
import typer
from hydra import compose, initialize_config_dir
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def _load_config(config_name: str = "default_config"):
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


def _class_metadata(metadata: dict, target: str) -> tuple[int, dict[int, str]]:
    if target == "class":
        mapping = metadata["class_to_idx"]
    elif target == "disease":
        mapping = metadata["disease_to_idx"]
    elif target == "plant":
        mapping = metadata["plant_to_idx"]
    else:
        raise ValueError(f"Unsupported target '{target}'. Expected one of ['class', 'disease', 'plant'] for visualize.")

    idx_to_name = {idx: name for name, idx in mapping.items()}
    return len(mapping), idx_to_name


def visualize(
    model_checkpoint: str | Path | None = None,
    figure_name: str = "embeddings.png",
    target: str | None = None,
    data_dir: str | None = None,
    config_name: str = "default_config",
) -> None:
    """Visualize model predictions."""
    from src.plants.model import Model

    cfg = _load_config(config_name)
    hparams = cfg.experiments
    if target is None:
        target = hparams.target
    if data_dir is None:
        data_dir = cfg.dataloader.data_dir

    device = _select_device(hparams.device)
    base_dir = _repo_root()
    resolved_data_dir = _resolve_path(data_dir, base_dir)
    metadata_path = _resolve_path(hparams.metadata_path, base_dir)
    model_dir = _resolve_path(hparams.model_dir, base_dir)
    checkpoint_path = _resolve_path(model_checkpoint or model_dir / "model.pth", base_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    label_count, idx_to_name = _class_metadata(metadata, target)

    state = torch.load(checkpoint_path, map_location="cpu")
    state_num_classes = int(state["fc1.weight"].shape[0])
    state_in_channels = int(state["conv1.weight"].shape[1])
    if state_num_classes != label_count:
        print(
            f"Warning: checkpoint has {state_num_classes} classes, but metadata has {label_count}. "
            "Using checkpoint classes for embeddings."
        )
    if state_in_channels != int(cfg.model.in_channels):
        print(
            f"Warning: checkpoint expects {state_in_channels} input channel(s), but config has "
            f"{cfg.model.in_channels}. Using checkpoint channels for embeddings."
        )

    model: torch.nn.Module = Model(
        num_classes=state_num_classes,
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
    model.eval()
    model.fc1 = torch.nn.Identity()

    processed_dir = resolved_data_dir / "processed"
    val_images = torch.load(processed_dir / "val_images.pt")
    if target == "class":
        labels_path = processed_dir / "val_labels.pt"
    elif target == "disease":
        labels_path = processed_dir / "val_disease_labels.pt"
    elif target == "plant":
        labels_path = processed_dir / "val_plant_labels.pt"
    else:
        raise ValueError(f"Unsupported target '{target}'. Expected one of ['class', 'disease', 'plant'] for visualize.")
    val_labels = torch.load(labels_path)
    val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(val_dataset, batch_size=32):
            images, target = batch
            images = images.to(device)
            predictions = model(images)
            embeddings.append(predictions.cpu())
            targets.append(target.cpu())
        embeddings = torch.cat(embeddings).numpy()
        targets = torch.cat(targets).numpy()

    if embeddings.shape[1] > 500:  # Reduce dimensionality for large embeddings
        pca = PCA(n_components=100)
        embeddings = pca.fit_transform(embeddings)
    tsne = TSNE(n_components=2)
    embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(min(10, label_count)):
        mask = targets == i
        label = idx_to_name.get(int(i), str(i))
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=label)
    plt.legend()
    reports_dir = _resolve_path("reports/figures", base_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    fig_path = reports_dir / figure_name
    plt.savefig(fig_path)
    print(f"Visualization saved to {fig_path}")


def visualize_raw_data(
    data_dir: str = "data",
    split: str = "train",
    figure_name: str = "raw_sample_grid.png",
    sample_count: int = 9,
) -> dict[tuple[int, int, int], int]:
    """Visualize raw images (folders per label) and report shape consistency."""
    from src.plants.data import ALLOWED_EXTENSIONS

    reports_dir = Path("reports/figures")
    reports_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(data_dir)
    metadata_path = data_root / "processed" / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    split_hint = metadata.get("splits", {}).get(split)

    candidates = [
        Path(split_hint) if split_hint else None,
        data_root / "raw" / "PlantVillage" / split,
        data_root / "raw" / split,
    ]
    raw_split = next((p for p in candidates if p and p.exists()), None)
    if raw_split is None:
        raise FileNotFoundError(f"Could not locate raw split '{split}' under {data_root}")

    image_paths = [p for p in raw_split.rglob("*") if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
    if not image_paths:
        raise FileNotFoundError(f"No raw images found in {raw_split}")

    sample_paths = image_paths[: min(sample_count, len(image_paths))]
    shape_counts: dict[tuple[int, int, int], int] = {}

    cols = 3
    rows = (len(sample_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, img_path in zip(axes_flat, sample_paths, strict=False):
        with Image.open(img_path) as img:
            rgb = img.convert("RGB")
            h, w = rgb.height, rgb.width
            c = len(rgb.getbands())
            shape = (h, w, c)
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
            ax.imshow(rgb)
            ax.set_title(f"{img_path.parent.name}\n{h}x{w}x{c}", fontsize=8)
            ax.axis("off")

    for ax in axes_flat[len(sample_paths) :]:
        ax.axis("off")

    plt.tight_layout()
    fig_path = reports_dir / figure_name
    plt.savefig(fig_path)
    plt.close(fig)

    if len(shape_counts) == 1:
        sole = next(iter(shape_counts))
        typer.echo(f"Raw shapes consistent: {sole}")
    else:
        typer.echo(f"Raw shapes vary: {shape_counts}")
    typer.echo(f"Raw sample grid saved to {fig_path}")
    return shape_counts


def visualize_processed_data(
    processed_dir: str = "data/processed",
    split: str = "train",
    figure_name: str = "sample_grid.png",
    sample_count: int = 9,
) -> None:
    """Show processed sample grid and label distributions."""
    processed_path = Path(processed_dir)
    images_path = processed_path / f"{split}_images.pt"
    labels_path = processed_path / f"{split}_labels.pt"
    disease_labels_path = processed_path / f"{split}_disease_labels.pt"
    plant_labels_path = processed_path / f"{split}_plant_labels.pt"
    metadata_path = processed_path / "metadata.json"

    if not images_path.exists():
        raise FileNotFoundError(f"Missing processed split '{split}' in {processed_path}. Run preprocessing first.")

    reports_dir = Path("reports/figures")
    reports_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    class_to_idx = metadata.get("class_to_idx", {})
    disease_to_idx = metadata.get("disease_to_idx", {})
    plant_to_idx = metadata.get("plant_to_idx", {})
    mean_val = metadata.get("mean")
    std_val = metadata.get("std")
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    idx_to_disease = {idx: name for name, idx in disease_to_idx.items()}
    idx_to_plant = {idx: name for name, idx in plant_to_idx.items()}

    images = torch.load(images_path)
    class_labels = torch.load(labels_path)
    disease_labels = torch.load(disease_labels_path)
    plant_labels = torch.load(plant_labels_path)

    split_sizes: dict[str, int] = {}
    for sp in ("train", "val"):
        sp_path = processed_path / f"{sp}_images.pt"
        if sp_path.exists():
            split_sizes[sp] = torch.load(sp_path).shape[0]

    def _counts_to_named(
        unique_tensor: torch.Tensor, counts_tensor: torch.Tensor, mapping: dict[int, str]
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        for label_idx, count in zip(unique_tensor.tolist(), counts_tensor.tolist(), strict=False):
            name = mapping.get(label_idx, str(label_idx))
            result[name] = count
        return result

    class_unique, class_counts = torch.unique(class_labels, return_counts=True)
    disease_unique, disease_counts = torch.unique(disease_labels, return_counts=True)
    plant_unique, plant_counts = torch.unique(plant_labels, return_counts=True)

    def _plot_distribution(labels, counts, mapping, title, filename):
        names = [mapping.get(int(idx), str(int(idx))) for idx in labels.tolist()]
        counts_list = counts.tolist()
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(names)), counts_list)
        plt.xticks(range(len(names)), names, rotation=90, fontsize=7)
        plt.title(title)
        plt.tight_layout()
        dist_path = reports_dir / filename
        plt.savefig(dist_path)
        plt.close()
        print(f"Saved {title} to {dist_path}")

    typer.echo(f"Processed directory: {processed_path}")
    typer.echo(f"Split sizes: {split_sizes}")
    typer.echo(f"Image shape: {tuple(images.shape[1:])}")
    typer.echo(f"Classes ({len(class_to_idx)}): {_counts_to_named(class_unique, class_counts, idx_to_class)}")
    typer.echo(f"Diseases ({len(disease_to_idx)}): {_counts_to_named(disease_unique, disease_counts, idx_to_disease)}")
    typer.echo(f"Plants ({len(plant_to_idx)}): {_counts_to_named(plant_unique, plant_counts, idx_to_plant)}")

    _plot_distribution(
        class_unique, class_counts, idx_to_class, f"{split} class distribution", f"{split}_class_dist.png"
    )
    _plot_distribution(
        disease_unique, disease_counts, idx_to_disease, f"{split} disease distribution", f"{split}_disease_dist.png"
    )
    _plot_distribution(
        plant_unique, plant_counts, idx_to_plant, f"{split} plant distribution", f"{split}_plant_dist.png"
    )

    sample_count = min(sample_count, images.shape[0])
    sample_indices = torch.linspace(0, images.shape[0] - 1, steps=sample_count, dtype=torch.long)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for ax, idx in zip(axes.flat, sample_indices.tolist(), strict=False):
        img = images[idx].permute(1, 2, 0)  # (H, W, C)
        if mean_val is not None and std_val is not None:
            img = img * float(std_val) + float(mean_val)
        img = torch.clamp(img, 0.0, 1.0).cpu()
        class_name = idx_to_class.get(int(class_labels[idx]), str(int(class_labels[idx])))
        disease_name = idx_to_disease.get(int(disease_labels[idx]), str(int(disease_labels[idx])))
        plant_name = idx_to_plant.get(int(plant_labels[idx]), str(int(plant_labels[idx])))
        ax.imshow(img.numpy())
        ax.set_title(f"{plant_name}\n{class_name}\nDisease: {disease_name}", fontsize=8)
        ax.axis("off")

    for ax in axes.flat[sample_count:]:
        ax.axis("off")

    plt.tight_layout()
    fig_path = reports_dir / figure_name
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Sample grid saved to {fig_path}")


def main(
    source: Annotated[
        str, typer.Option("--source", "-s", help="Choose 'processed' or 'raw' visualization")
    ] = "processed",
    split: Annotated[str, typer.Option("--split", help="Dataset split to visualize")] = "train",
    figure_name: Annotated[
        Optional[str],  # noqa: UP007
        typer.Option("--figure-name", "-f", help="Optional output filename"),
    ] = None,
    sample_count: Annotated[int, typer.Option("--sample-count", "-n", help="Number of samples to show in grid")] = 9,
    data_dir: Annotated[
        Optional[str],  # noqa: UP007
        typer.Option("--data-dir", help="Data root for raw images"),
    ] = None,
    processed_dir: Annotated[
        Optional[str],  # noqa: UP007
        typer.Option("--processed-dir", help="Directory with processed tensors"),
    ] = None,
    target: Annotated[
        Optional[str],  # noqa: UP007
        typer.Option("--target", help="Override target for embeddings visualization"),
    ] = None,
    model_checkpoint: Annotated[
        Optional[Path],  # noqa: UP007
        typer.Option("--model-checkpoint", help="Model checkpoint for embeddings"),
    ] = None,
    config_name: Annotated[str, typer.Option("--config-name", help="Hydra config name to load.")] = "default_config",
) -> None:
    """Dispatch to raw or processed visualization based on a single flag."""
    cfg = _load_config(config_name)
    if data_dir is None:
        data_dir = cfg.dataloader.data_dir
    resolved_data_dir = str(_resolve_path(data_dir, _repo_root()))
    if processed_dir is None:
        processed_dir = str(Path(resolved_data_dir) / "processed")

    normalized = source.lower()
    if normalized == "raw":
        visualize_raw_data(
            data_dir=resolved_data_dir,
            split=split,
            figure_name=figure_name or "raw_sample_grid.png",
            sample_count=sample_count,
        )
    elif normalized == "processed":
        visualize_processed_data(
            processed_dir=processed_dir,
            split=split,
            figure_name=figure_name or "sample_grid.png",
            sample_count=sample_count,
        )
    elif normalized == "embeddings":
        visualize(
            model_checkpoint=model_checkpoint,
            figure_name=figure_name or "embeddings.png",
            target=target,
            data_dir=resolved_data_dir,
            config_name=config_name,
        )
    else:
        raise typer.BadParameter("source must be either 'raw', 'processed', or 'embeddings'")


if __name__ == "__main__":
    _ensure_repo_root_on_path()
    typer.run(main)
