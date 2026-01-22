"""Locust performance scenarios for the FastAPI inference service.

Run with:
  locust -f tests/performancetests/locustfile.py --host=http://localhost:8000
"""

import io
import os

from locust import HttpUser, between, task

# Tiny 1x1 red PNG (binary literal to avoid file IO).
SAMPLE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xcf"
    b"\xc0\x00\x00\x04\xfe\x01\xfeA\xae\xb2\x9c\x00\x00\x00\x00IEND\xaeB`\x82"
)


class RandomInferenceUser(HttpUser):
    """Hits lightweight endpoints (/, /predict-random) to measure latency."""

    wait_time = between(0.5, 2)

    @task(3)
    def homepage(self) -> None:
        self.client.get("/")

    @task(2)
    def predict_random(self) -> None:
        self.client.post("/predict-random")


class UploadInferenceUser(HttpUser):
    """Sends image uploads to /predict to exercise heavier path."""

    wait_time = between(1, 3)

    @task
    def predict_with_upload(self) -> None:
        file_tuple = (
            "sample.png",
            io.BytesIO(SAMPLE_PNG),
            "image/png",
        )
        # Allow overriding target via env (default FastAPI expects /predict).
        endpoint = os.environ.get("LOCUST_PREDICT_PATH", "/predict")
        self.client.post(endpoint, files={"file": file_tuple})
