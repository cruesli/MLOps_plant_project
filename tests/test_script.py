import os
import subprocess
import zipfile
from pathlib import Path

import pytest


@pytest.mark.slow
def test_get_data_script():
    """Test the get_data.sh script."""
    # Check for Kaggle credentials
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json_path.exists() and ("KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ):
        pytest.skip("Kaggle credentials not found. Skipping test.")

    dataset = "mohitsingh1804/plantvillage"
    outdir = Path("data/raw")
    zip_path = outdir / "plantvillage.zip"

    outdir.mkdir(parents=True, exist_ok=True)

    # Download
    subprocess.run(
        ["uv", "run", "kaggle", "datasets", "download", "-d", dataset, "-p", str(outdir)],
        check=True,
    )

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(outdir)

    # Check if the data directory is created and not empty
    assert outdir.exists()
    assert any(outdir.iterdir())
