import io

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torchvision import transforms

from src.plants import api


@pytest.fixture()
def demo_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Provide a TestClient with a stubbed model context to avoid GCS downloads."""

    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.num_classes = num_classes

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            batch = x.shape[0]
            return torch.zeros((batch, self.num_classes))

    # Build a tiny in-memory demo image.
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    class_names = ["class_0", "class_1"]
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    ctx = api.ModelContext(
        model=DummyModel(len(class_names)),
        device=torch.device("cpu"),
        class_names=class_names,
        transform=transform,
        demo_images={"sample.png": (img_bytes, "class_1")},
        metrics={"final_accuracy": 0.9},
        target="class",
    )

    # Patch model loading so startup uses the stub context.
    monkeypatch.setattr(api, "_load_model_context", lambda: ctx)
    api.context = ctx

    return TestClient(api.app)


def test_index_returns_html(demo_client: TestClient) -> None:
    resp = demo_client.get("/")
    assert resp.status_code == 200
    assert "Plant Classifier" in resp.text


def test_predict_random_returns_demo_sample(demo_client: TestClient) -> None:
    resp = demo_client.post("/predict-random")
    assert resp.status_code == 200

    data = resp.json()
    assert data["sample"] == "sample.png"
    assert data["true_label"] == "Class 1"
    assert data["target"] == "class"
    assert data["metrics"]["final_accuracy"] == 0.9
    assert isinstance(data["top_predictions"], list)
    assert len(data["top_predictions"]) > 0


def test_demo_endpoint_serves_image(demo_client: TestClient) -> None:
    resp = demo_client.get("/demo/sample.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content.startswith(b"\x89PNG")
