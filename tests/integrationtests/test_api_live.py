import os

import pytest
import requests


@pytest.fixture(scope="session")
def base_url() -> str:
    """Base URL for the live API (set via MYENDPOINT env var)."""
    endpoint = os.environ.get("MYENDPOINT")
    if not endpoint:
        pytest.skip("Set MYENDPOINT to run live API integration tests.")
    return endpoint.rstrip("/")


def _request(method: str, url: str, *, timeout: int = 30) -> requests.Response:
    """Request helper with basic retries for cold starts."""
    last_exc: Exception | None = None
    for _ in range(3):
        try:
            return requests.request(method, url, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    raise RuntimeError("Request failed without exception.")


def test_index_returns_html(base_url: str) -> None:
    resp = _request("GET", f"{base_url}/", timeout=45)
    assert resp.status_code == 200
    assert "Plant Classifier" in resp.text


def test_predict_random_returns_payload(base_url: str) -> None:
    resp = _request("POST", f"{base_url}/predict-random", timeout=60)
    assert resp.status_code == 200
    data = resp.json()
    assert "top_predictions" in data
    assert "image_url" in data
    assert "target" in data
    assert data["top_predictions"]


def test_demo_endpoint_serves_image(base_url: str) -> None:
    random_resp = _request("POST", f"{base_url}/predict-random", timeout=60)
    assert random_resp.status_code == 200
    image_url = random_resp.json()["image_url"]
    resp = _request("GET", f"{base_url}{image_url}", timeout=30)
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG")
