#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch
from PIL import Image


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _normalize_gcs_prefix(prefix: str) -> str:
    return prefix[5:] if prefix.startswith("gs://") else prefix


def _download_gcs(gcs_path: str, local_path: Path) -> None:
    try:
        from gcsfs import GCSFileSystem
    except ImportError as exc:
        raise RuntimeError("gcsfs is required to download data from GCS.") from exc

    fs = GCSFileSystem()
    if not fs.exists(gcs_path):
        raise FileNotFoundError(f"Missing GCS object: {gcs_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fs.get(gcs_path, str(local_path))


def _upload_gcs(local_path: Path, gcs_path: str) -> None:
    try:
        from gcsfs import GCSFileSystem
    except ImportError as exc:
        raise RuntimeError("gcsfs is required to upload data to GCS.") from exc

    fs = GCSFileSystem()
    fs.put(str(local_path), gcs_path)


def _load_metadata(metadata_path: Path) -> dict:
    return json.loads(metadata_path.read_text())


def _label_mapping(metadata: dict, target: str) -> dict[int, str]:
    if target == "class":
        mapping = metadata["class_to_idx"]
    elif target == "disease":
        mapping = metadata["disease_to_idx"]
    elif target == "plant":
        mapping = metadata["plant_to_idx"]
    else:
        raise ValueError(f"Unsupported target '{target}'. Expected one of ['class', 'disease', 'plant'].")

    return {int(idx): name for name, idx in mapping.items()}


def _label_tensor_path(target: str) -> str:
    if target == "class":
        return "val_labels.pt"
    if target == "disease":
        return "val_disease_labels.pt"
    if target == "plant":
        return "val_plant_labels.pt"
    raise ValueError(f"Unsupported target '{target}'. Expected one of ['class', 'disease', 'plant'].")


def _tensor_to_image(tensor: torch.Tensor, mean: float, std: float) -> Image.Image:
    # Tensor is normalized; undo and clamp to [0, 1]
    img = tensor.clone()
    img = img * std + mean
    img = img.clamp(0, 1)
    img = (img * 255).to(torch.uint8)
    img = img.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(img)


def main() -> int:
    data_gcs = _require_env("PLANTS_DATA_GCS")
    demo_gcs = os.environ.get("PLANTS_DEMO_GCS", "gs://mlops-plants/demo")
    target = os.environ.get("PLANTS_TARGET", "class")
    sample_count = int(os.environ.get("PLANTS_DEMO_COUNT", "20"))

    base = _normalize_gcs_prefix(data_gcs.rstrip("/"))
    demo_base = _normalize_gcs_prefix(demo_gcs.rstrip("/"))

    workdir = Path("/tmp/demo_samples")
    workdir.mkdir(parents=True, exist_ok=True)

    metadata_path = workdir / "metadata.json"
    _download_gcs(f"{base}/metadata.json", metadata_path)
    metadata = _load_metadata(metadata_path)

    images_path = workdir / "val_images.pt"
    labels_path = workdir / _label_tensor_path(target)
    _download_gcs(f"{base}/val_images.pt", images_path)
    _download_gcs(f"{base}/{_label_tensor_path(target)}", labels_path)

    images = torch.load(images_path)
    labels = torch.load(labels_path)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("Images and labels count do not match.")

    mean = float(metadata.get("mean", 0.0))
    std = float(metadata.get("std", 1.0))
    idx_to_label = _label_mapping(metadata, target)

    indices = list(range(images.shape[0]))
    random.shuffle(indices)
    indices = indices[:sample_count]

    labels_manifest: dict[str, str] = {}
    for i, idx in enumerate(indices):
        img = _tensor_to_image(images[idx], mean, std)
        label_name = idx_to_label.get(int(labels[idx]), "unknown")
        filename = f"sample_{i:02d}.png"
        local_file = workdir / filename
        img.save(local_file)
        labels_manifest[filename] = label_name
        _upload_gcs(local_file, f"{demo_base}/{filename}")

    manifest_path = workdir / "labels.json"
    manifest_path.write_text(json.dumps(labels_manifest, indent=2))
    _upload_gcs(manifest_path, f"{demo_base}/labels.json")

    print(f"Uploaded {len(indices)} samples to gs://{demo_base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
