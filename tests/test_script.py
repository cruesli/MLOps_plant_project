import os
import subprocess
from pathlib import Path
import zipfile

import pytest

@pytest.mark.slow
def test_get_data_script():
    """Test the get_data.sh script."""
    # Check for Kaggle credentials
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json_path.exists() and ("KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ):
        pytest.skip("Kaggle credentials not found. Skipping test.")

    DATASET = "mohitsingh1804/plantvillage"
    OUTDIR = Path("data/raw")
    ZIP = OUTDIR / "plantvillage.zip"

    OUTDIR.mkdir(parents=True, exist_ok=True)

    # Download
    subprocess.run(
        ["uv", "run", "kaggle", "datasets", "download", "-d", DATASET, "-p", str(OUTDIR)],
        check=True,
    )

    # Extract
    with zipfile.ZipFile(ZIP, "r") as zip_ref:
        zip_ref.extractall(OUTDIR)
    
    # Check if the data directory is created and not empty
    assert OUTDIR.exists()
    assert any(OUTDIR.iterdir())
