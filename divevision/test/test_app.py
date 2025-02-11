import io
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from divevision.src.app.main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200


def test_upload_image():
    random_image = Image.fromarray(np.random.randn(128, 128, 3), mode="RGB")
    buffer = io.BytesIO()
    random_image.save(buffer, "PNG")
    response = client.post(
        "/image/",
        files={"file": ("foo.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 200
