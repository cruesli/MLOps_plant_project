# plants

MLOps project for plant disease classification using the PlantVillage dataset.

## Project overview

This repository implements a reproducible training and deployment pipeline for plant disease classification. The focus is on operational reliability (configs, tracking, CI/CD, containerization) rather than maximizing model accuracy.

Key components:
- Training pipeline built around a custom CNN baseline in `src/plants/model.py`.
- Hydra configuration for experiments and sweeps under `configs/`.
- DVC-tracked data artifacts (`data.dvc`) with GCS storage via `dvc-gs`.
- W&B logging for training runs.
- FastAPI inference service in `src/plants/api.py`.
- Cloud Build configs for training/sweeps and API deployment in `cloudbuild/`.

## Data

The PlantVillage dataset is not included in the repo. See `Dataset.md` for instructions.

A helper script downloads the dataset from Kaggle:

```bash
uv run ./scripts/get_data.sh
```

This places data under `data/raw` and is then processed into tensors by `src/plants/data.py`.

## Dependencies

We use Python 3.13 and `uv` with dependencies defined in `pyproject.toml`:

```bash
uv sync --dev
```

If you prefer pip, a minimal runtime set is available in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Local usage

From the repo root:

1) Preprocess data
```bash
uv run src/plants/data.py
```

2) Train
```bash
uv run src/plants/train.py
```

3) Evaluate
```bash
uv run src/plants/evaluate.py
```

4) Visualize
```bash
uv run src/plants/visualize.py
```

5) Run the API locally
```bash
uv run src/plants/api.py
```

## Cloud workflows

Cloud Build configuration files live in `cloudbuild/`:
- `cloudbuild/cloudbuild.yaml` submits a Vertex AI training job.
- `cloudbuild/cloudbuild.sweep.yaml` runs a sweep workflow.
- `cloudbuild/cloudbuild.api.yaml` builds and deploys the API container to Cloud Run.

Cloud Run service URL:
```txt
gcloud run services describe plant-api --region europe-west1 --format='value(status.url)'
```

Dockerfiles are stored in `dockerfiles/`:
- `dockerfiles/train.dockerfile`
- `dockerfiles/api.dockerfile`

## Project structure

```txt
├── .devcontainer/            # Devcontainer setup
├── .dvc/                     # DVC metadata
├── .github/                  # GitHub Actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       ├── linting.yaml
│       ├── pre-commit-update.yaml
│       └── tests.yaml
├── cloudbuild/               # Cloud Build configs
│   ├── cloudbuild.api.yaml
│   ├── cloudbuild.sweep.yaml
│   └── cloudbuild.yaml
├── configs/                  # Hydra configs
│   ├── dataloader/
│   ├── experiments/
│   ├── model/
│   ├── sweeps/
│   └── default_config.yaml
├── data/                     # Local data (raw/processed)
├── dockerfiles/              # Dockerfiles
│   ├── api.dockerfile
│   └── train.dockerfile
├── docs/                     # MkDocs site
│   ├── mkdocs.yaml
│   └── source/
│       └── index.md
├── models/                   # Saved model artifacts (local)
├── notebooks/                # Exploration notebooks
├── outputs/                  # Local run outputs
├── reports/                  # Report template and figures
│   ├── README.md
│   ├── report.py
│   └── figures/
├── wandb/                    # W&B run artifacts (local)
├── scripts/                  # Helper scripts
│   ├── create_demo_samples.py
│   ├── get_data.sh
│   └── promote_best.py
├── src/                      # Source code
│   └── plants/
│       ├── api.py
│       ├── data.py
│       ├── evaluate.py
│       ├── model.py
│       ├── train.py
│       └── visualize.py
├── tests/                    # Tests
│   ├── apitests/
│   ├── integrationtests/
│   ├── performancetests/
│   ├── conftest.py
│   └── test_*.py
├── Dataset.md
├── README.md
├── data.dvc                
├── pyproject.toml
├── requirements.txt
├── tasks.py
└── uv.lock
```

## Docs

MkDocs configuration lives in `docs/mkdocs.yaml`.

```bash
uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build
uv run mkdocs serve --config-file docs/mkdocs.yaml
```

## Notes

This repository was originally created with the `mlops_template` cookiecutter and has been extended with `uv`, DVC, Cloud Build, and Cloud Run deployment workflows.
