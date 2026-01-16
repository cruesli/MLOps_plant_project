import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from src.plants.model import Model
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.plants.data import ALLOWED_EXTENSIONS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def visualize(
    model_checkpoint: str | Path = "models/model.pth",
    figure_name: str = "embeddings.png",
) -> None:
    """Visualize model predictions."""
    checkpoint_path = Path(model_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model: torch.nn.Module = Model()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(DEVICE)
    model.eval()
    model.fc = torch.nn.Identity()

    val_images = torch.load("data/processed/val_images.pt")
    val_labels = torch.load("data/processed/val_labels.pt")
    val_dataset = torch.utils.data.TensorDataset(val_images, val_labels)

    embeddings, targets = [], []
    with torch.inference_mode():
        for batch in torch.utils.data.DataLoader(val_dataset, batch_size=32):
            images, target = batch
            images = images.to(DEVICE)
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
    for i in range(10):
        mask = targets == i
        plt.scatter(embeddings[mask, 0], embeddings[mask, 1], label=str(i))
    plt.legend()
    plt.savefig(f"reports/figures/{figure_name}")
    print(f"Visualization saved to {figure_name}")


def visualize_raw_data(
    data_dir: str = "data",
    split: str = "train",
    figure_name: str = "raw_sample_grid.png",
    sample_count: int = 9,
) -> dict[tuple[int, int, int], int]:
    """Visualize raw images (folders per label) and report shape consistency."""
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
    source: str = typer.Option("processed", "--source", "-s", help="Choose 'processed' or 'raw' visualization"),
    split: str = typer.Option("train", "--split", help="Dataset split to visualize"),
    figure_name: str | None = typer.Option(None, "--figure-name", "-f", help="Optional output filename"),
    sample_count: int = typer.Option(9, "--sample-count", "-n", help="Number of samples to show in grid"),
    data_dir: str = typer.Option("data", "--data-dir", help="Data root for raw images"),
    processed_dir: str = typer.Option("data/processed", "--processed-dir", help="Directory with processed tensors"),
) -> None:
    """Dispatch to raw or processed visualization based on a single flag."""
    normalized = source.lower()
    if normalized == "raw":
        visualize_raw_data(
            data_dir=data_dir,
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
    else:
        raise typer.BadParameter("source must be either 'raw' or 'processed'")


if __name__ == "__main__":
    typer.run(main)
