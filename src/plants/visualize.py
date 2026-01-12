import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import typer
from model import Model
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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


def visualize_data(
    processed_dir: str = "data/processed",
    split: str = "train",
    figure_name: str = "sample_grid.png",
) -> None:
    """Show a 3x3 grid of sample images and print basic dataset stats."""
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
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    idx_to_disease = {idx: name for name, idx in disease_to_idx.items()}
    idx_to_plant = {idx: name for name, idx in plant_to_idx.items()}

    images = torch.load(images_path)
    class_labels = torch.load(labels_path)
    disease_labels = torch.load(disease_labels_path)
    plant_labels = torch.load(plant_labels_path)

    # General statistics
    split_sizes: dict[str, int] = {}
    for sp in ("train", "val"):
        sp_path = processed_path / f"{sp}_images.pt"
        if sp_path.exists():
            split_sizes[sp] = torch.load(sp_path).shape[0]

    def _counts_to_named(unique_tensor: torch.Tensor, counts_tensor: torch.Tensor, mapping: dict[int, str]) -> dict[str, int]:
        result: dict[str, int] = {}
        for label_idx, count in zip(unique_tensor.tolist(), counts_tensor.tolist()):
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

    _plot_distribution(class_unique, class_counts, idx_to_class, f"{split} class distribution", f"{split}_class_dist.png")
    _plot_distribution(
        disease_unique, disease_counts, idx_to_disease, f"{split} disease distribution", f"{split}_disease_dist.png"
    )
    _plot_distribution(plant_unique, plant_counts, idx_to_plant, f"{split} plant distribution", f"{split}_plant_dist.png")

    # 3x3 grid of samples
    sample_count = min(9, images.shape[0])
    sample_indices = torch.linspace(0, images.shape[0] - 1, steps=sample_count, dtype=torch.long)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for ax, idx in zip(axes.flat, sample_indices.tolist()):
        img = images[idx].squeeze()
        class_name = idx_to_class.get(int(class_labels[idx]), str(int(class_labels[idx])))
        disease_name = idx_to_disease.get(int(disease_labels[idx]), str(int(disease_labels[idx])))
        plant_name = idx_to_plant.get(int(plant_labels[idx]), str(int(plant_labels[idx])))
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{plant_name}\n{class_name}\nDisease: {disease_name}", fontsize=8)
        ax.axis("off")

    for ax in axes.flat[sample_count:]:
        ax.axis("off")

    plt.tight_layout()
    fig_path = reports_dir / figure_name
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"Sample grid saved to {fig_path}")

if __name__ == "__main__":
    typer.run(visualize_data)