from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from torchvision import transforms

from src.plants.model import Model

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_ROOT / "models" / "model.pth"
DEFAULT_METADATA_PATH = APP_ROOT / "data" / "processed" / "metadata.json"
DEFAULT_TARGET = os.getenv("PLANT_TARGET", "class")


@dataclass
class Job:
    task: str
    status: str = "queued"
    returncode: int | None = None
    output: list[str] = field(default_factory=list)


def _append_output(job: Job, line: str) -> None:
    job.output.append(line.rstrip())
    if len(job.output) > 500:
        job.output = job.output[-500:]


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found at {path}")
    with path.open() as handle:
        return json.load(handle)


def _label_map(metadata: dict[str, Any], target: str) -> list[str]:
    if target == "class":
        mapping = metadata["class_to_idx"]
    elif target == "disease":
        mapping = metadata["disease_to_idx"]
    elif target == "plant":
        mapping = metadata["plant_to_idx"]
    else:
        raise ValueError("target must be one of: class, disease, plant")
    return [name for name, _ in sorted(mapping.items(), key=lambda item: item[1])]


@asynccontextmanager
async def lifespan(app: FastAPI):
    metadata = _load_metadata(DEFAULT_METADATA_PATH)
    labels = _label_map(metadata, DEFAULT_TARGET)
    num_classes = len(labels)

    model = Model(
        num_classes=num_classes,
        in_channels=3,
        conv1_out=32,
        conv1_kernel=3,
        conv1_stride=1,
        conv2_out=64,
        conv2_kernel=3,
        conv2_stride=2,
        conv2_padding=1,
        dropout=0.2,
    )
    if not DEFAULT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {DEFAULT_MODEL_PATH}")
    state = torch.load(DEFAULT_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    mean = metadata.get("mean", 0.0)
    std = metadata.get("std", 1.0)
    mean_tensor = torch.tensor(float(mean))
    std_tensor = torch.tensor(float(std))
    if torch.isclose(std_tensor, torch.tensor(0.0)):
        std_tensor = torch.tensor(1.0)

    preprocess = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )

    app.state.model = model
    app.state.labels = labels
    app.state.mean = mean_tensor
    app.state.std = std_tensor
    app.state.preprocess = preprocess
    app.state.target = DEFAULT_TARGET

    yield


app = FastAPI(lifespan=lifespan)
app.state.jobs: dict[str, Job] = {}
app.state.jobs_lock = threading.Lock()


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Plant MLOps Control Panel</title>
    <style>
      :root {
        --bg: #f7f2e8;
        --panel: #fff7ec;
        --ink: #1e1b16;
        --muted: #6b645b;
        --accent: #2f6b3f;
        --accent-dark: #214d2d;
        --line: #1e1b16;
      }
      * {
        box-sizing: border-box;
      }
      body {
        margin: 0;
        font-family: "Georgia", "Times New Roman", serif;
        color: var(--ink);
        background: radial-gradient(circle at top, #f9f6ee 0%, #efe5d6 45%, #e8f0e6 100%);
        min-height: 100vh;
        padding: 24px;
      }
      .shell {
        max-width: 980px;
        margin: 0 auto;
        display: grid;
        gap: 20px;
      }
      header {
        border: 2px solid var(--line);
        background: var(--panel);
        padding: 18px 20px;
        box-shadow: 8px 8px 0 var(--line);
      }
      h1 {
        margin: 0 0 8px;
        font-size: 28px;
        letter-spacing: 0.4px;
      }
      p {
        margin: 0;
        color: var(--muted);
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 16px;
      }
      .card {
        border: 2px solid var(--line);
        background: #fffdf7;
        padding: 16px;
        box-shadow: 6px 6px 0 var(--line);
      }
      .card h2 {
        margin: 0 0 8px;
        font-size: 18px;
      }
      .card p {
        font-size: 14px;
        margin: 0 0 12px;
      }
      button {
        width: 100%;
        border: 2px solid var(--line);
        background: var(--accent);
        color: #fff;
        padding: 10px 12px;
        font-size: 16px;
        cursor: pointer;
      }
      button.secondary {
        background: #f0e8db;
        color: var(--ink);
      }
      button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }
      .output {
        border: 2px solid var(--line);
        background: #0e0d0b;
        color: #f2f0eb;
        padding: 12px;
        min-height: 180px;
        font-family: "Courier New", Courier, monospace;
        white-space: pre-wrap;
        overflow: auto;
      }
      .progress {
        height: 14px;
        border: 2px solid var(--line);
        background: #f0e8db;
        overflow: hidden;
      }
      .progress-bar {
        height: 100%;
        width: 0%;
        background: var(--accent);
        transition: width 0.4s ease;
      }
      .progress-bar.running {
        background: repeating-linear-gradient(
          45deg,
          #2f6b3f,
          #2f6b3f 12px,
          #3f7f4f 12px,
          #3f7f4f 24px
        );
        animation: move 1s linear infinite;
      }
      .progress-bar.error {
        background: #9c2f2f;
      }
      @keyframes move {
        from {
          background-position: 0 0;
        }
        to {
          background-position: 48px 0;
        }
      }
      .row {
        display: grid;
        gap: 12px;
      }
      .upload {
        display: grid;
        gap: 10px;
      }
      input[type="file"] {
        padding: 6px;
        border: 2px solid var(--line);
        background: #fffdf7;
      }
      .tag {
        display: inline-block;
        padding: 2px 8px;
        border: 1px solid var(--line);
        font-size: 12px;
        margin-top: 6px;
        background: #f0e8db;
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <header>
        <h1>Plant MLOps Control Panel</h1>
        <p>Run preprocessing, training, evaluation, and visualization directly from the project.</p>
      </header>

      <section class="grid">
        <div class="card">
          <h2>Download Data</h2>
          <p>Fetch PlantVillage data using Kaggle credentials.</p>
          <button onclick="runTask('download')" id="btn-download">Download</button>
        </div>
        <div class="card">
          <h2>Preprocess</h2>
          <p>Convert raw images into tensors and metadata.</p>
          <button onclick="runTask('preprocess')" id="btn-preprocess">Preprocess</button>
        </div>
        <div class="card">
          <h2>Train</h2>
          <p>Run the default training configuration.</p>
          <button onclick="runTask('train')" id="btn-train">Train</button>
        </div>
        <div class="card">
          <h2>Evaluate</h2>
          <p>Evaluate the latest checkpoint.</p>
          <button onclick="runTask('evaluate')" id="btn-evaluate">Evaluate</button>
        </div>
        <div class="card">
          <h2>Visualize</h2>
          <p>Create report figures in reports/figures.</p>
          <button onclick="runTask('visualize')" id="btn-visualize">Visualize</button>
        </div>
        <div class="card">
          <h2>Predict</h2>
          <p>Upload a leaf image to get a prediction.</p>
          <div class="upload">
            <input id="predict-file" type="file" accept="image/*" />
            <button class="secondary" onclick="runPredict()">Predict</button>
            <span class="tag" id="predict-result">No prediction yet.</span>
          </div>
        </div>
      </section>

      <section class="row">
        <div class="card">
          <h2>Run Output</h2>
          <div class="row">
            <div class="progress">
              <div id="progress-bar" class="progress-bar"></div>
            </div>
            <div id="status" class="tag">Idle</div>
          </div>
          <div id="output" class="output">Ready.</div>
        </div>
      </section>
    </div>

    <script>
      const output = document.getElementById("output");
      const statusTag = document.getElementById("status");
      const progressBar = document.getElementById("progress-bar");
      const buttons = ["download", "preprocess", "train", "evaluate", "visualize"]
        .map((name) => document.getElementById(`btn-${name}`));
      let poller = null;

      function setBusy(state) {
        buttons.forEach((btn) => (btn.disabled = state));
      }

      function setProgress(state) {
        progressBar.classList.remove("running", "error");
        if (state === "running") {
          progressBar.classList.add("running");
          progressBar.style.width = "100%";
          statusTag.textContent = "Running";
        } else if (state === "failed") {
          progressBar.classList.add("error");
          progressBar.style.width = "100%";
          statusTag.textContent = "Failed";
        } else if (state === "success") {
          progressBar.style.width = "100%";
          statusTag.textContent = "Complete";
        } else {
          progressBar.style.width = "0%";
          statusTag.textContent = "Idle";
        }
      }

      async function pollStatus(jobId) {
        if (poller) clearInterval(poller);
        poller = setInterval(async () => {
          const response = await fetch(`/status/${jobId}`);
          const data = await response.json();
          output.textContent = data.output || "No output.";
          if (data.status === "running") {
            setProgress("running");
            return;
          }
          clearInterval(poller);
          poller = null;
          setBusy(false);
          setProgress(data.status === "success" ? "success" : "failed");
        }, 1000);
      }

      async function runTask(task) {
        setBusy(true);
        output.textContent = `Running ${task}...`;
        setProgress("running");
        try {
          const response = await fetch(`/run/${task}`, { method: "POST" });
          const data = await response.json();
          if (!response.ok) {
            output.textContent = data.error || "Task failed to start.";
            setProgress("failed");
            setBusy(false);
            return;
          }
          pollStatus(data.job_id);
        } catch (err) {
          output.textContent = `Request failed: ${err}`;
          setProgress("failed");
          setBusy(false);
        } finally {
          if (!poller) {
            setBusy(false);
          }
        }
      }

      async function runPredict() {
        const fileInput = document.getElementById("predict-file");
        const result = document.getElementById("predict-result");
        if (!fileInput.files.length) {
          result.textContent = "Pick an image first.";
          return;
        }
        const form = new FormData();
        form.append("file", fileInput.files[0]);
        result.textContent = "Predicting...";
        try {
          const response = await fetch("/predict", { method: "POST", body: form });
          const data = await response.json();
          if (!response.ok) {
            result.textContent = data.detail || "Prediction failed.";
            return;
          }
          result.textContent = `${data.label} (${(data.score * 100).toFixed(1)}%)`;
        } catch (err) {
          result.textContent = `Error: ${err}`;
        }
      }
    </script>
  </body>
</html>
"""


def _run_command_background(job_id: str, command: list[str]) -> None:
    with app.state.jobs_lock:
        job = app.state.jobs[job_id]
        job.status = "running"

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        with app.state.jobs_lock:
            job = app.state.jobs[job_id]
            job.status = "failed"
            job.returncode = 127
            _append_output(job, f"Command failed: {exc}")
        return

    assert process.stdout is not None
    for line in process.stdout:
        with app.state.jobs_lock:
            job = app.state.jobs[job_id]
            _append_output(job, line)

    returncode = process.wait()
    with app.state.jobs_lock:
        job = app.state.jobs[job_id]
        job.returncode = returncode
        job.status = "success" if returncode == 0 else "failed"


@app.post("/run/{task}")
def run_task(task: str):
    commands = {
        "download": ["uv", "run", "./scripts/get_data.sh"],
        "preprocess": ["uv", "run", "src/plants/data.py"],
        "train": ["uv", "run", "src/plants/train.py"],
        "evaluate": ["uv", "run", "src/plants/evaluate.py"],
        "visualize": ["uv", "run", "src/plants/visualize.py"],
    }
    if task not in commands:
        raise HTTPException(status_code=404, detail="Unknown task.")
    job_id = uuid.uuid4().hex
    with app.state.jobs_lock:
        app.state.jobs[job_id] = Job(task=task)
    thread = threading.Thread(target=_run_command_background, args=(job_id, commands[task]), daemon=True)
    thread.start()
    return JSONResponse(status_code=202, content={"job_id": job_id})


@app.get("/status/{job_id}")
def job_status(job_id: str):
    with app.state.jobs_lock:
        job = app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return {
            "task": job.task,
            "status": job.status,
            "returncode": job.returncode,
            "output": "\n".join(job.output),
        }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):  # noqa: B008
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    try:
        image = Image.open(file.file).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    tensor = app.state.preprocess(image)
    tensor = (tensor - app.state.mean) / app.state.std
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        logits = app.state.model(tensor)
        probs = torch.softmax(logits, dim=1)
        score, idx = torch.max(probs, dim=1)

    label = app.state.labels[int(idx.item())]
    return {
        "label": label,
        "score": float(score.item()),
        "target": app.state.target,
    }
