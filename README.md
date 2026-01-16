# plants

MLOps project on plant image classification

## Overall Goal of the Project: 

The primary objective of this project is to develop a robust MLOps pipeline for automated plant disease detection using the PlantVillage dataset. As MLOps engineers at a start-up, our priority is not the absolute performance of the model, but the speed and reliability of the pipeline itself. 
We aim to create a system where experiments are fully reproducible through structured configuration and experiment tracking. To achieve this, we will implement a baseline model and moving toward a tuned version. A part of our operational goal is to containerize our application using Docker to ensure that our training and environments are consistent across different machines, which is essential for a MLOps workflow.



## Data: Source and Initial Scope: 

**Dataset:** The PlantVillage dataset is not included in this repository. See [`Dataset.md`](Dataset.md) for download and setup instructions.

We will utilize the PlantVillage dataset from Kaggle, which contains thousands of images of various plant leaves in both healthy and diseased states. To keep the project manageable and avoid complex data-loading issues, we will initially work with a subset of the data. We will develop a get_data.sh script to handle the initial downloading ensuring that our data pipeline is as automated as possible.



## Models and Frameworks: 

We will use PyTorch as our primary machine learning framework. To streamline development and optimize our code, we plan to use the following:

    ResNet-18: We will use this as our baseline model because it is a smaller, efficient architecture that allows for faster iteration.

    Hydra: This will manage our configuration files and hyperparameters, allowing us to run multiple experiment variations easily.

    Weights & Biases (W&B): We will use this to log our training progress, metrics, and model artifacts, ensuring that all team members can monitor the project's status in real-time.

By focusing on these tools, we aim to move quickly through the design phase and focus on the long-term operations and reliability of the system.


## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── plants/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Getting Started

### Installation

1. **Clone the repository**
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

To run the project locally, you can use the provided scripts in `src/plants/`.

**1. Download Data**
The dataset is not included in the repo. Download it using the helper script (requires Kaggle API credentials):
```bash
uv run ./scripts/get_data.sh
```

**2. Preprocess Data**
Process the raw images into tensors for training:
```bash
uv run src/plants/data.py
```

**3. Train Model**
Train the model. You can run the default hyperparameters from the config files using the following:
```bash
uv run src/plants/train.py
```

Or you can specify what configuration you want to try from the configs directory:
```bash
uv run src/plants/train.py experiments=XXX dataloader=XXX model=XXX
```

**4. Evaluate**
Evaluate the just trained model:
```bash
uv run src/plants/evaluate.py
```
Or another model:
```bash
uv run src/plants/evaluate.py model=XXX
```

**5. Visualize**
Generate data distribution plots and sample grids in `reports/figures/`:
```bash
uv run src/plants/visualize.py
```
