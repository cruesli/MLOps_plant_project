import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torchvision import transforms

from src.plants import api as api_module


def _make_png_bytes(color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    image = Image.new("RGB", (2, 2), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class _DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)


def _dummy_context() -> api_module.ModelContext:
    transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]
    )
    demo_bytes = _make_png_bytes()
    return api_module.ModelContext(
        model=_DummyModel(),
        device=torch.device("cpu"),
        class_names=["class_0", "class_1", "class_2"],
        transform=transform,
        demo_images={"demo.png": (demo_bytes, "apple___healthy")},
        metrics={"run_id": "unit-test"},
        target="class",
    )


def _client() -> TestClient:
    api_module.context = None
    api_module._load_model_context = _dummy_context  # type: ignore[assignment]
    return TestClient(api_module.app)


@pytest.fixture
def client() -> TestClient:
    with _client() as client_instance:
        yield client_instance


def test_root_returns_html(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Plant Classifier" in response.text


def test_predict_random(client: TestClient) -> None:
    response = client.post("/predict-random")
    assert response.status_code == 200
    payload = response.json()
    assert payload["top_predictions"]
    assert payload["image_url"].endswith("demo.png")
    assert payload["target"] == "class"


def test_demo_image(client: TestClient) -> None:
    response = client.get("/demo/demo.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_refresh_model(client: TestClient) -> None:
    response = client.post("/refresh")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
