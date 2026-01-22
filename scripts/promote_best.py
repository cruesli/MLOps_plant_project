#!/usr/bin/env python3
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import wandb


@dataclass
class Settings:
    entity: str
    project: str
    sweep_id: str
    metric: str
    model_gcs: str


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _normalize_gcs_prefix(prefix: str) -> str:
    return prefix[5:] if prefix.startswith("gs://") else prefix


def _best_run(settings: Settings) -> wandb.apis.public.Run:
    api = wandb.Api()
    sweep = api.sweep(f"{settings.entity}/{settings.project}/{settings.sweep_id}")
    if not sweep.runs:
        raise RuntimeError("Sweep has no runs.")
    best = None
    best_value = None
    for run in sweep.runs:
        summary = run.summary or {}
        value = summary.get(settings.metric)
        if value is None:
            continue
        if best is None or value > best_value:
            best = run
            best_value = value
    if best is None:
        raise RuntimeError(f"No runs with metric '{settings.metric}' found.")
    return best


def _copy_gcs(src: str, dst: str) -> None:
    try:
        from gcsfs import GCSFileSystem
    except ImportError as exc:
        raise RuntimeError("gcsfs is required to copy models in GCS.") from exc

    fs = GCSFileSystem()
    if src.startswith("gs://"):
        if not fs.exists(src):
            raise FileNotFoundError(f"Missing source object: {src}")
        fs.copy(src, dst)
        return

    local_path = Path(src)
    if not local_path.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    fs.put(str(local_path), dst)


def main() -> int:
    settings = Settings(
        entity=_require_env("WANDB_ENTITY"),
        project=_require_env("WANDB_PROJECT"),
        sweep_id=_require_env("WANDB_SWEEP_ID"),
        metric=os.environ.get("WANDB_METRIC", "final_accuracy"),
        model_gcs=_require_env("PLANTS_MODEL_GCS"),
    )

    best = _best_run(settings)
    base = _normalize_gcs_prefix(settings.model_gcs.rstrip("/"))
    src = f"gs://{base}/{best.id}/model.pth"
    dst = f"gs://{base}/best/model.pth"

    _copy_gcs(src, dst)

    metrics_path = f"gs://{base}/best/metrics.json"
    metrics = {
        "run_id": best.id,
        "metric": settings.metric,
        "metric_value": best.summary.get(settings.metric),
        "final_accuracy": best.summary.get("final_accuracy"),
        "final_precision": best.summary.get("final_precision"),
        "final_recall": best.summary.get("final_recall"),
        "final_f1": best.summary.get("final_f1"),
    }
    local_metrics = Path("/tmp/best_metrics.json")
    local_metrics.write_text(json.dumps(metrics, indent=2))
    _copy_gcs(str(local_metrics), metrics_path)

    print(f"Promoted best run {best.id} ({settings.metric}={best.summary.get(settings.metric)})")
    print(f"Copied {src} -> {dst}")
    print(f"Wrote metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
