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
    filenames = [
        "lsui_sample_19.jpg",
        "lsui_sample_1226.jpg",
    ]

    filepaths = []
    for filename in filenames:
        filepaths.append(Path(prefix + filename).resolve())

    assert all(map(lambda p: p.exists(), filepaths))

    response = client.post(
        "/images/",
        files=[
            (
                "files",
                open(filepath, "rb"),
            )
            for filepath in filepaths
        ],
    )
    assert response.status_code == 200
