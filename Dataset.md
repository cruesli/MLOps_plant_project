# Dataset: PlantVillage

## Overview

This project uses the **PlantVillage** dataset, a publicly available image dataset for plant disease classification. The dataset contains labeled images of healthy and diseased plant leaves across multiple crop species.

The dataset is **not stored in this repository** due to its size (~850 MB) and is instead downloaded locally by each developer using the Kaggle API.

---

## Source

- **Provider:** Kaggle

- **Dataset:** PlantVillage
- **Kaggle URL:** [https://www.kaggle.com/datasets/mohitsingh1804/plantvillage](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)
- **Kaggle Slug:** `mohitsingh1804/plantvillage`

---

## Licensing and Usage

This dataset is publicly available via Kaggle. Users are responsible for reviewing and complying with the dataset’s license and Kaggle’s terms of use before downloading or using the data.

The dataset files themselves are **not redistributed** as part of this repository.

---

## How to Obtain the Dataset

### Prerequisites

- A Kaggle account
- Kaggle API credentials (`kaggle.json`) configured locally
- Python environment managed with `uv`

### Setup Kaggle Credentials

Create a file at: `~/.kaggle/kaggle.json`

with the following contents:

```json
{
  "username": "<your_kaggle_username>",
  "key": "<your_kaggle_api_key>"
}
```

On macOS/Linux, ensure correct permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Download and Extract

From the repository root, run: `./scripts/get_data.sh`

This will:

- Download the dataset ZIP from Kaggle

- Extract it locally into the data/raw/ directory

- The data/ directory is ignored by Git and exists only on local machines.

## Expected Directory Structure

After extraction, the local structure should look similar to:

```txt
data/
└── raw/
    ├── PlantVillage/
    │   ├── Apple___Apple_scab/
    │   ├── Apple___Black_rot/
    │   ├── ...
    │   └── Tomato___healthy/
    └── plantvillage.zip
```

Exact class names and counts are defined by the dataset provider.

## Versioning and Reproducibility

- The dataset is referenced by its Kaggle slug

- If Kaggle updates the dataset, results may change

- Any experiments should record the dataset download date and commit hash of the code used
