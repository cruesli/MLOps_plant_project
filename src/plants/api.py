from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_repo_root_on_path() -> None:
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_config(config_name: str = "default_config") -> DictConfig:
    config_dir = _repo_root() / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name)


def _normalize_gcs_prefix(prefix: str) -> str:
    return prefix[5:] if prefix.startswith("gs://") else prefix


def _download_gcs_file(gcs_path: str, local_path: Path) -> None:
    try:
        from gcsfs import GCSFileSystem
    except ImportError as exc:  # pragma: no cover - only needed in cloud jobs
        raise RuntimeError("gcsfs is required to download models from GCS.") from exc

    fs = GCSFileSystem()
    if not fs.exists(gcs_path):
        raise FileNotFoundError(f"Missing GCS object: {gcs_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fs.get(gcs_path, str(local_path))


def _maybe_fetch_metadata(metadata_path: Path) -> dict:
    if metadata_path.exists():
        return json.loads(metadata_path.read_text())

    data_gcs = os.environ.get("PLANTS_DATA_GCS")
    if not data_gcs:
        raise FileNotFoundError(f"Metadata not found at {metadata_path} and PLANTS_DATA_GCS is not set.")

    base = _normalize_gcs_prefix(data_gcs.rstrip("/"))
    candidates = [f"{base}/metadata.json"]
    if base.endswith("/processed"):
        parent = base.rsplit("/", 1)[0]
        candidates.extend([f"{parent}/metadata.json", f"{parent}/processed/metadata.json"])
    else:
        candidates.append(f"{base}/processed/metadata.json")

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            _download_gcs_file(candidate, metadata_path)
            return json.loads(metadata_path.read_text())
        except FileNotFoundError as exc:
            last_error = exc

    if last_error:
        raise last_error

    return json.loads(metadata_path.read_text())


def _class_metadata(metadata: dict, target: str) -> tuple[int, list[str]]:
    if target == "class":
        mapping = metadata["class_to_idx"]
    elif target == "disease":
        mapping = metadata["disease_to_idx"]
    elif target == "plant":
        mapping = metadata["plant_to_idx"]
    else:
        raise ValueError(f"Unsupported target '{target}'. Expected one of ['class', 'disease', 'plant'] for inference.")

    class_names = [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]
    return len(mapping), class_names


def _format_label(label: str) -> str:
    if label.startswith("class_") and label[6:].isdigit():
        return f"Class {label[6:]}"
    if "___" in label:
        parts = [part.replace("_", " ").strip() for part in label.split("___", 1)]
        return " · ".join(part.title() for part in parts)
    return label.replace("_", " ").strip().title()


@dataclass
class ModelContext:
    model: torch.nn.Module
    device: torch.device
    class_names: list[str]
    transform: transforms.Compose
    demo_images: dict[str, tuple[bytes, str]]
    metrics: dict
    target: str


def _load_model_context() -> ModelContext:
    _ensure_repo_root_on_path()
    cfg = _load_config()
    hparams = cfg.experiments
    target = os.environ.get("PLANTS_TARGET", "class")

    base_dir = _repo_root()
    metadata_path = base_dir / hparams.metadata_path
    metadata = _maybe_fetch_metadata(metadata_path)
    num_classes, class_names = _class_metadata(metadata, target)

    model_gcs = os.environ.get("PLANTS_MODEL_GCS")
    if not model_gcs:
        raise RuntimeError("PLANTS_MODEL_GCS is required to load the best model from GCS.")
    model_tag = os.environ.get("PLANTS_MODEL_TAG", "best")
    model_key = f"{_normalize_gcs_prefix(model_gcs.rstrip('/'))}/{model_tag}/model.pth"
    local_model = Path("/tmp") / f"model_{model_tag}.pth"
    _download_gcs_file(model_key, local_model)

    state = torch.load(local_model, map_location="cpu")
    state_num_classes = int(state["fc.weight"].shape[0])
    state_in_channels = int(state["layer1.0.weight"].shape[1])
    if state_num_classes != num_classes:
        num_classes = state_num_classes
        class_names = [f"class_{idx}" for idx in range(num_classes)]

    from src.plants.model import Model

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
    model.eval()

    device = torch.device("cpu")
    model.to(device)

    mean_val = metadata.get("mean", 0.0)
    std_val = metadata.get("std", 1.0)
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean_val] * 3, std=[std_val] * 3),
        ]
    )

    demo_images = _load_demo_images()
    metrics = _load_best_metrics()

    return ModelContext(
        model=model,
        device=device,
        class_names=class_names,
        transform=transform,
        demo_images=demo_images,
        metrics=metrics,
        target=target,
    )


def _load_demo_images() -> dict[str, tuple[bytes, str]]:
    demo_gcs = os.environ.get("PLANTS_DEMO_GCS", "gs://mlops-plants/demo")
    base = _normalize_gcs_prefix(demo_gcs.rstrip("/"))
    workdir = Path("/tmp/demo")
    workdir.mkdir(parents=True, exist_ok=True)

    labels_path = workdir / "labels.json"
    _download_gcs_file(f"{base}/labels.json", labels_path)
    labels = json.loads(labels_path.read_text())

    demo_images: dict[str, tuple[bytes, str]] = {}
    for filename, label in labels.items():
        local_file = workdir / filename
        _download_gcs_file(f"{base}/{filename}", local_file)
        demo_images[filename] = (local_file.read_bytes(), label)
    if not demo_images:
        raise RuntimeError("No demo images found in PLANTS_DEMO_GCS.")
    return demo_images


def _load_best_metrics() -> dict:
    model_gcs = os.environ.get("PLANTS_MODEL_GCS")
    if not model_gcs:
        return {}
    base = _normalize_gcs_prefix(model_gcs.rstrip("/"))
    metrics_path = Path("/tmp/best_metrics.json")
    try:
        _download_gcs_file(f"{base}/best/metrics.json", metrics_path)
    except FileNotFoundError:
        return {}
    return json.loads(metrics_path.read_text())


app = FastAPI(title="Plant Disease Inference")
context: ModelContext | None = None


@app.on_event("startup")
def _startup() -> None:
    global context
    context = _load_model_context()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Plant Inference</title>
    <style>
      :root {
        --bg: #0f172a;
        --card: #0b1220;
        --accent: #22d3ee;
        --muted: #94a3b8;
        --text: #e2e8f0;
        --pill: #1f2937;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Space Grotesk", "IBM Plex Sans", "Segoe UI", sans-serif;
        background: radial-gradient(1200px 600px at 20% -10%, #1e293b 0%, #0f172a 55%);
        color: var(--text);
      }
      .wrap { max-width: 980px; margin: 40px auto; padding: 0 20px; }
      .hero { display: grid; gap: 18px; }
      .title { font-size: 32px; font-weight: 700; }
      .subtitle { color: var(--muted); }
      .grid { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }
      .card {
        background: var(--card);
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
      }
      button {
        background: var(--accent);
        color: #0b1220;
        border: none;
        padding: 12px 18px;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
      }
      button:hover { filter: brightness(0.95); }
      .pill {
        display: inline-flex;
        padding: 6px 10px;
        border-radius: 999px;
        background: var(--pill);
        color: var(--muted);
        font-size: 12px;
      }
      .img {
        width: 100%;
        border-radius: 12px;
        border: 1px solid #1f2937;
        background: #0b1220;
      }
      .label { font-weight: 600; }
      .metric { display: grid; gap: 6px; }
      .metric-row { display: flex; justify-content: space-between; color: var(--muted); font-size: 14px; }
      .pred { display: grid; gap: 10px; }
      .bar {
        height: 8px;
        border-radius: 999px;
        background: #1f2937;
        overflow: hidden;
      }
      .bar > span {
        display: block;
        height: 100%;
        background: var(--accent);
      }
      .note { color: var(--muted); font-size: 12px; margin-top: 8px; }
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="hero card">
        <div class="title">Plant Classifier</div>
        <div class="subtitle">Random sample inference using the best model found in the sweep.</div>
        <div class="pill" id="target-pill">Target: class</div>
        <button onclick="runInference()">Random Inference</button>
      </div>

      <div class="grid" style="margin-top: 20px;">
        <div class="card">
          <div class="label">Sample</div>
          <p class="note" id="sample-name">Waiting for inference...</p>
          <img id="image" class="img" />
          <p class="note" id="true-label"></p>
        </div>

        <div class="card">
          <div class="label">Top Predictions</div>
          <div id="predictions" class="pred"></div>
        </div>

        <div class="card">
          <div class="label">Best Model Metrics</div>
          <div id="metrics" class="metric"></div>
          <div class="note">Metrics are from the best model run (sweep) and update after re‑promotion.</div>
        </div>
      </div>
    </div>
    <script>
      const fmt = (v) => (v === null || v === undefined) ? "—" : Number(v).toFixed(4);

      async function runInference() {
        const response = await fetch("/predict-random", { method: "POST" });
        const data = await response.json();
        if (data.image_url) {
          document.getElementById("image").src = data.image_url;
        }
        if (data.target) {
          document.getElementById("target-pill").textContent = "Target: " + data.target;
        }
        document.getElementById("sample-name").textContent = data.sample || "";
        document.getElementById("true-label").textContent = data.true_label ? `True label: ${data.true_label}` : "";

        const predEl = document.getElementById("predictions");
        predEl.innerHTML = "";
        (data.top_predictions || []).forEach((p) => {
          const row = document.createElement("div");
          row.innerHTML = `
            <div class="metric-row"><span>${p.label}</span><span>${fmt(p.probability)}</span></div>
            <div class="bar"><span style="width: ${Math.round(p.probability * 100)}%"></span></div>
          `;
          predEl.appendChild(row);
        });

        const metricsEl = document.getElementById("metrics");
        metricsEl.innerHTML = "";
        const m = data.metrics || {};
        const rows = [
          ["Run ID", m.run_id],
          ["Primary metric", `${m.metric || "—"} (${fmt(m.metric_value)})`],
          ["Accuracy", fmt(m.final_accuracy)],
          ["Precision", fmt(m.final_precision)],
          ["Recall", fmt(m.final_recall)],
          ["F1", fmt(m.final_f1)],
        ];
        rows.forEach(([k, v]) => {
          const row = document.createElement("div");
          row.className = "metric-row";
          row.innerHTML = `<span>${k}</span><span>${v || "—"}</span>`;
          metricsEl.appendChild(row);
        });
      }
    </script>
  </body>
</html>
"""


@app.post("/predict-random")
def predict_random() -> JSONResponse:
    if context is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
    if not context.demo_images:
        return JSONResponse({"error": "No demo images available"}, status_code=500)

    filename = random.choice(list(context.demo_images.keys()))
    image_bytes, true_label = context.demo_images[filename]
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = context.transform(image).unsqueeze(0).to(context.device)

    with torch.inference_mode():
        logits = context.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    topk = min(5, probs.shape[0])
    values, indices = torch.topk(probs, topk)
    results = [
        {"label": _format_label(context.class_names[idx]), "probability": float(val)}
        for val, idx in zip(values.tolist(), indices.tolist(), strict=False)
    ]

    payload = {
        "sample": filename,
        "true_label": _format_label(true_label),
        "top_predictions": results,
        "metrics": context.metrics,
        "image_url": f"/demo/{filename}",
        "target": context.target,
    }
    return JSONResponse(payload)


@app.get("/demo/{filename}")
def demo_image(filename: str) -> Response:
    if context is None or filename not in context.demo_images:
        return Response(status_code=404)
    image_bytes, _ = context.demo_images[filename]
    return Response(content=image_bytes, media_type="image/png")


@app.post("/refresh")
def refresh_model() -> JSONResponse:
    global context
    context = _load_model_context()
    return JSONResponse({"status": "ok", "message": "Model reloaded from GCS."})


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
