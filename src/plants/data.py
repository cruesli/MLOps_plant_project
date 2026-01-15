from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import multiprocessing as mp
import torch
import typer
from PIL import Image
from torch.utils.data import Dataset, TensorDataset
from torchvision import transforms

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional at runtime
    tqdm = None

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class SplitData:
    """Container for processed split tensors."""

    images: torch.Tensor
    class_labels: torch.Tensor
    disease_labels: torch.Tensor
    plant_labels: torch.Tensor


class MyDataset(Dataset):
    """Dataset helper for PlantVillage images."""

    def __init__(self, data_path: Path | str, split: str | None = None, target: str = "class") -> None:
        # Allow passing either the repository root, data directory, or raw/processed directly.
        data_path = Path(data_path)
        if data_path.name in {"raw", "processed"}:
            self.data_root = data_path.parent
        else:
            self.data_root = data_path

        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        self.split = split
        self.target = target

        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),          # keeps 3 channels if input is RGB
            ]
        )


        self.images: torch.Tensor | None = None
        self.class_labels: torch.Tensor | None = None
        self.disease_labels: torch.Tensor | None = None
        self.plant_labels: torch.Tensor | None = None

        if self.split:
            self._load_processed_split(self.split, target=self.target)

    def __len__(self) -> int:
        if self.images is None:
            return 0
        return len(self.images)

    def __getitem__(self, index: int):
        if self.images is None or self.class_labels is None or self.disease_labels is None:
            raise RuntimeError("Processed data not loaded. Run preprocess() or pass a split to MyDataset.")

        if self.target == "class":
            return self.images[index], self.class_labels[index]
        if self.target == "disease":
            return self.images[index], self.disease_labels[index]
        if self.target == "plant":
            return self.images[index], self.plant_labels[index]
        if self.target == "both":
            return self.images[index], self.class_labels[index], self.disease_labels[index]
        if self.target == "all":
            return (
                self.images[index],
                self.class_labels[index],
                self.disease_labels[index],
                self.plant_labels[index],
            )
        msg = f"Unknown target '{self.target}'. Expected one of ['class', 'disease', 'both']."
        raise ValueError(msg)

    @staticmethod
    def _disease_index(label_name: str, disease_map: dict[str, int], allow_new: bool) -> int:
        """Return consistent disease index; 0 is always reserved for healthy leaves."""
        disease_key = label_name.split("___", 1)[1] if "___" in label_name else label_name
        disease_key = disease_key.replace("_", " ").lower()
        if "healthy" in disease_key:
            return disease_map["healthy"]

        if disease_key not in disease_map:
            if not allow_new:
                msg = f"Found unseen disease label '{label_name}' in validation split."
                raise ValueError(msg)
            disease_map[disease_key] = len(disease_map)
        return disease_map[disease_key]

    @staticmethod
    def _plant_index(label_name: str) -> str:
        """Return the plant type portion of a class label."""
        plant_key = label_name.split("___", 1)[0]
        return plant_key.replace("_", " ").lower()

    def _locate_split_dir(self, aliases: Iterable[str]) -> Path:
        """Find a split directory by trying aliases and then globbing under raw/."""
        for alias in aliases:
            candidate = self.raw_dir / alias
            if candidate.is_dir():
                return candidate

        for alias in aliases:
            matches = [p for p in self.raw_dir.glob(f"**/{alias}") if p.is_dir()]
            if matches:
                return matches[0]

        alias_list = ", ".join(aliases)
        msg = f"Could not find any of [{alias_list}] under {self.raw_dir}"
        raise FileNotFoundError(msg)

    def _iter_image_files(self, root: Path) -> Iterable[Path]:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
                yield path

    def _load_and_transform_split(
        self,
        split_dir: Path,
        class_to_idx: dict[str, int],
        disease_to_idx: dict[str, int],
        plant_to_idx: dict[str, int],
        allow_new_classes: bool = True,
        show_progress: bool = True,
    ) -> SplitData:
        images: list[torch.Tensor] = []
        class_labels: list[int] = []
        disease_labels: list[int] = []
        plant_labels: list[int] = []

        class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if not class_dirs:
            msg = f"No class folders found in {split_dir}"
            raise ValueError(msg)

        progress = None
        if show_progress and tqdm is not None:
            total_images = sum(1 for class_dir in class_dirs for _ in self._iter_image_files(class_dir))
            progress = tqdm(total=total_images, desc=f"Processing {split_dir.name}", unit="img")

        for class_dir in sorted(class_dirs):
            label_name = class_dir.name
            if label_name not in class_to_idx:
                if not allow_new_classes:
                    msg = f"Found unseen class '{label_name}' in validation split."
                    raise ValueError(msg)
                class_to_idx[label_name] = len(class_to_idx)
            class_idx = class_to_idx[label_name]
            disease_idx = self._disease_index(label_name, disease_to_idx, allow_new_classes)
            plant_key = self._plant_index(label_name)
            if plant_key not in plant_to_idx:
                if not allow_new_classes:
                    msg = f"Found unseen plant '{plant_key}' in validation split."
                    raise ValueError(msg)
                plant_to_idx[plant_key] = len(plant_to_idx)
            plant_idx = plant_to_idx[plant_key]

            for image_path in self._iter_image_files(class_dir):
                with Image.open(image_path) as img:
                    tensor_img = self.transform(img.convert("RGB"))
                images.append(tensor_img)
                class_labels.append(class_idx)
                disease_labels.append(disease_idx)
                plant_labels.append(plant_idx)
                if progress is not None:
                    progress.update(1)

        if progress is not None:
            progress.close()

        if not images:
            msg = f"No images found in split directory {split_dir}"
            raise ValueError(msg)

        stacked_images = torch.stack(images)
        class_tensor = torch.tensor(class_labels, dtype=torch.long)
        disease_tensor = torch.tensor(disease_labels, dtype=torch.long)
        plant_tensor = torch.tensor(plant_labels, dtype=torch.long)
        return SplitData(stacked_images, class_tensor, disease_tensor, plant_tensor)

    @staticmethod
    def normalize(
        images: torch.Tensor, *, mean: torch.Tensor | float | None = None, std: torch.Tensor | float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize images using provided or computed statistics."""
        mean_tensor = images.mean() if mean is None else mean
        std_tensor = images.std() if std is None else std

        if not torch.is_tensor(mean_tensor):
            mean_tensor = torch.tensor(mean_tensor, device=images.device)
        if not torch.is_tensor(std_tensor):
            std_tensor = torch.tensor(std_tensor, device=images.device)

        if torch.isclose(std_tensor, torch.tensor(0.0, device=std_tensor.device)):
            std_tensor = torch.tensor(1.0, device=std_tensor.device)

        normalized = (images - mean_tensor) / std_tensor
        return normalized, mean_tensor, std_tensor

    def preprocess(self, *, show_progress: bool = True) -> None:
        """Process PlantVillage images into train/val tensors."""
        if not self.raw_dir.exists():
            msg = f"Raw data directory not found: {self.raw_dir}. Run ./scripts/get_data.sh first."
            raise FileNotFoundError(msg)

        self.processed_dir.mkdir(parents=True, exist_ok=True)

        train_dir = self._locate_split_dir(("train", "training"))
        val_dir = self._locate_split_dir(("val", "valid", "validation"))

        class_to_idx: dict[str, int] = {}
        disease_to_idx: dict[str, int] = {"healthy": 0}
        plant_to_idx: dict[str, int] = {}

        train_split = self._load_and_transform_split(
            train_dir,
            class_to_idx,
            disease_to_idx,
            plant_to_idx,
            allow_new_classes=True,
            show_progress=show_progress,
        )
        val_split = self._load_and_transform_split(
            val_dir,
            class_to_idx,
            disease_to_idx,
            plant_to_idx,
            allow_new_classes=False,
            show_progress=show_progress,
        )

        train_images, mean, std = self.normalize(train_split.images)
        val_images, _, _ = self.normalize(val_split.images, mean=mean, std=std)

        torch.save(train_images, self.processed_dir / "train_images.pt")
        torch.save(train_split.class_labels, self.processed_dir / "train_labels.pt")
        torch.save(train_split.disease_labels, self.processed_dir / "train_disease_labels.pt")
        torch.save(train_split.plant_labels, self.processed_dir / "train_plant_labels.pt")

        torch.save(val_images, self.processed_dir / "val_images.pt")
        torch.save(val_split.class_labels, self.processed_dir / "val_labels.pt")
        torch.save(val_split.disease_labels, self.processed_dir / "val_disease_labels.pt")
        torch.save(val_split.plant_labels, self.processed_dir / "val_plant_labels.pt")

        metadata = {
            "class_to_idx": class_to_idx,
            "disease_to_idx": disease_to_idx,
            "plant_to_idx": plant_to_idx,
            "mean": float(mean),
            "std": float(std),
            "splits": {"train": str(train_dir), "val": str(val_dir)},
        }
        (self.processed_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"Processed data saved to {self.processed_dir}")

    def _load_processed_split(self, split: str, target: str | None = None) -> None:
        """Load processed tensors for a split and prepare __getitem__ to serve them."""
        split = split.lower()
        images_path = self.processed_dir / f"{split}_images.pt"
        class_labels_path = self.processed_dir / f"{split}_labels.pt"
        disease_labels_path = self.processed_dir / f"{split}_disease_labels.pt"
        plant_labels_path = self.processed_dir / f"{split}_plant_labels.pt"

        if not images_path.exists():
            msg = f"Missing processed split '{split}' in {self.processed_dir}. Run preprocess() first."
            raise FileNotFoundError(msg)

        self.images = torch.load(images_path)
        self.class_labels = torch.load(class_labels_path)
        self.disease_labels = torch.load(disease_labels_path)
        self.plant_labels = torch.load(plant_labels_path)
        if target:
            self.target = target

    def load_plantvillage(self, target: str | None = None) -> tuple[TensorDataset, TensorDataset]:
        """Return processed train and val datasets.

        Args:
            target: Which target to use. Options:
                - "class": class label (plant + disease)
                - "disease": health/disease index (0=healthy, 1..=disease types)
                - "plant": plant type index
                - "both": returns (image, class_label, disease_label)
                - "all": returns (image, class_label, disease_label, plant_label)
        """
        target = target or self.target
        train_images = torch.load(self.processed_dir / "train_images.pt")
        train_labels = torch.load(self.processed_dir / "train_labels.pt")
        train_disease = torch.load(self.processed_dir / "train_disease_labels.pt")
        train_plant = torch.load(self.processed_dir / "train_plant_labels.pt")

        val_images = torch.load(self.processed_dir / "val_images.pt")
        val_labels = torch.load(self.processed_dir / "val_labels.pt")
        val_disease = torch.load(self.processed_dir / "val_disease_labels.pt")
        val_plant = torch.load(self.processed_dir / "val_plant_labels.pt")

        if target == "class":
            train_ds = TensorDataset(train_images, train_labels)
            val_ds = TensorDataset(val_images, val_labels)
        elif target == "disease":
            train_ds = TensorDataset(train_images, train_disease)
            val_ds = TensorDataset(val_images, val_disease)
        elif target == "plant":
            train_ds = TensorDataset(train_images, train_plant)
            val_ds = TensorDataset(val_images, val_plant)
        elif target == "both":
            train_ds = TensorDataset(train_images, train_labels, train_disease)
            val_ds = TensorDataset(val_images, val_labels, val_disease)
        elif target == "all":
            train_ds = TensorDataset(train_images, train_labels, train_disease, train_plant)
            val_ds = TensorDataset(val_images, val_labels, val_disease, val_plant)
        else:
            msg = f"Unknown target '{target}'. Expected one of ['class', 'disease', 'plant', 'both', 'all']."
            raise ValueError(msg)
        return train_ds, val_ds

def preprocess(data_path: Path = Path("data"), show_progress: bool = True) -> None:
    """CLI entrypoint for preprocessing PlantVillage data."""
    print("Preprocessing PlantVillage data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(show_progress=show_progress)

def main():
    typer.run(preprocess)

if __name__ == "__main__":
    mp.set_start_method("fork") 
    main()
