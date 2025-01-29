from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from divevision.src.app.main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome!"}


def test_upload_image():
    prefix = "divevision/test/resources/"
    filename = "lsui_sample_19.jpg"

    filepath = Path(prefix + filename).resolve()
    assert filepath.exists() and filepath.is_file()

    response = client.post(
        "/image/",
        files={"file": (filename, filepath.read_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
