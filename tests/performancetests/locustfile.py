"""Locust performance scenarios for the FastAPI inference service.

Run with:
  locust -f tests/performancetests/locustfile.py --host=http://localhost:8000
"""

from locust import HttpUser, between, task


class RandomInferenceUser(HttpUser):
    """Hits lightweight endpoints (/, /predict-random) to measure latency."""

    wait_time = between(0.5, 2)

    @task(3)
    def homepage(self) -> None:
        self.client.get("/")

    @task(2)
    def predict_random(self) -> None:
        self.client.post("/predict-random")


class RandomOnlyUser(HttpUser):
    """Optional user that only hits lightweight endpoints."""

    wait_time = between(0.5, 2)

    @task(3)
    def homepage(self) -> None:
        self.client.get("/")

    @task(2)
    def predict_random(self) -> None:
        self.client.post("/predict-random")
