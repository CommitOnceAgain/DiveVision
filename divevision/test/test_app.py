import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from divevision.src.app import supabase_api
from divevision.src.app.main import app

client = TestClient(app)

AUTH_HEADERS = {"Authorization": "Bearer access", "X-Refresh-Token": "refresh"}


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200


def test_signup(monkeypatch):
    monkeypatch.setattr(supabase_api, "create_user", lambda mail, password: True)
    response = client.post(
        "/signup/", json={"email": "test@example.com", "password": "123456"}
    )
    assert response.status_code == 201


def test_signup_failure(monkeypatch):
    monkeypatch.setattr(supabase_api, "create_user", lambda mail, password: False)
    response = client.post(
        "/signup/", json={"email": "test@example.com", "password": "123456"}
    )
    assert response.status_code == 400


def test_login(monkeypatch):
    monkeypatch.setattr(
        supabase_api,
        "sign_in_user",
        lambda mail, password: {
            "access_token": "access",
            "refresh_token": "refresh",
            "user_id": "user-123",
        },
    )
    response = client.post(
        "/login/", json={"email": "test@example.com", "password": "123456"}
    )
    assert response.status_code == 200
    assert response.json() == {
        "access_token": "access",
        "refresh_token": "refresh",
        "user_id": "user-123",
    }


def test_login_invalid_credentials(monkeypatch):
    monkeypatch.setattr(supabase_api, "sign_in_user", lambda mail, password: None)
    response = client.post(
        "/login/", json={"email": "test@example.com", "password": "wrong"}
    )
    assert response.status_code == 401


def test_upload_image_requires_auth_headers():
    random_image = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    random_image.save(buffer, "PNG")
    response = client.post(
        "/image/",
        files={"file": ("foo.png", buffer.getvalue(), "image/png")},
    )
    assert response.status_code == 422


def test_upload_image_persists_original_and_processed(monkeypatch):
    random_image = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    random_image.save(buffer, "PNG")

    uploads = []
    monkeypatch.setattr(
        supabase_api,
        "upload_image",
        lambda access_token, refresh_token, file, bucket, path=None: uploads.append(
            (bucket, path)
        )
        or (path or f"user-123/{bucket}.jpg"),
    )
    monkeypatch.setattr(
        supabase_api,
        "create_photo",
        lambda access_token, refresh_token, original_path, model_name: "photo-1",
    )
    completed = []
    monkeypatch.setattr(
        supabase_api,
        "mark_photo_completed",
        lambda access_token, refresh_token, photo_id, processed_path: completed.append(
            (photo_id, processed_path)
        )
        or True,
    )

    response = client.post(
        "/image/",
        files={"file": ("foo.png", buffer.getvalue(), "image/png")},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    # Original uploaded first (no explicit path), processed reuses that path.
    assert uploads[0] == (supabase_api.IMAGES_BUCKET, None)
    original_path = f"user-123/{supabase_api.IMAGES_BUCKET}.jpg"
    assert uploads[1] == (supabase_api.PROCESSED_IMAGES_BUCKET, original_path)
    assert completed == [("photo-1", original_path)]


def test_upload_image_marks_failed_when_processed_upload_fails(monkeypatch):
    random_image = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    random_image.save(buffer, "PNG")

    # This test only cares about the failure-path branching, not real model
    # output, so skip the (slow, CPU-only) real inference. `.load()` forces a
    # full decode while the source file is still open, same as the real
    # model's preprocessing step.
    def fake_predict(self, image):
        image.load()
        return [image]

    fake_model = type("FakeModel", (), {"name": "U-Shape", "predict": fake_predict})()
    monkeypatch.setattr(
        "divevision.src.app.main.UShapeModelWrapper", lambda: fake_model
    )

    def fake_upload(access_token, refresh_token, file, bucket, path=None):
        if bucket == supabase_api.IMAGES_BUCKET:
            return "user-123/original.jpg"
        return None

    monkeypatch.setattr(supabase_api, "upload_image", fake_upload)
    monkeypatch.setattr(
        supabase_api,
        "create_photo",
        lambda access_token, refresh_token, original_path, model_name: "photo-1",
    )
    failed = []
    monkeypatch.setattr(
        supabase_api,
        "mark_photo_failed",
        lambda access_token, refresh_token, photo_id: failed.append(photo_id) or True,
    )

    response = client.post(
        "/image/",
        files={"file": ("foo.png", buffer.getvalue(), "image/png")},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 502
    assert failed == ["photo-1"]


def test_delete_photo(monkeypatch):
    monkeypatch.setattr(
        supabase_api,
        "delete_photo",
        lambda access_token, refresh_token, photo_id: photo_id == "photo-1",
    )
    response = client.delete("/photos/photo-1/", headers=AUTH_HEADERS)
    assert response.status_code == 204


def test_delete_photo_not_found(monkeypatch):
    monkeypatch.setattr(
        supabase_api,
        "delete_photo",
        lambda access_token, refresh_token, photo_id: False,
    )
    response = client.delete("/photos/photo-1/", headers=AUTH_HEADERS)
    assert response.status_code == 404


def test_delete_account(monkeypatch):
    monkeypatch.setattr(
        supabase_api, "delete_account", lambda access_token, refresh_token: True
    )
    response = client.delete("/account/", headers=AUTH_HEADERS)
    assert response.status_code == 204


def test_delete_account_failure(monkeypatch):
    monkeypatch.setattr(
        supabase_api, "delete_account", lambda access_token, refresh_token: False
    )
    response = client.delete("/account/", headers=AUTH_HEADERS)
    assert response.status_code == 400


def test_leaderboard_rejects_missing_secret(monkeypatch):
    monkeypatch.setenv("LEADERBOARD_SHARED_SECRET", "top-secret")
    response = client.post(
        "/leaderboard/",
        json={
            "model_name": "U-Shape",
            "dataset_name": "UIEB",
            "metric_name": "ssim",
            "score": 0.9,
        },
    )
    assert response.status_code == 422  # missing required header


def test_leaderboard_rejects_wrong_secret(monkeypatch):
    monkeypatch.setenv("LEADERBOARD_SHARED_SECRET", "top-secret")
    response = client.post(
        "/leaderboard/",
        json={
            "model_name": "U-Shape",
            "dataset_name": "UIEB",
            "metric_name": "ssim",
            "score": 0.9,
        },
        headers={"X-Leaderboard-Secret": "wrong"},
    )
    assert response.status_code == 401


def test_leaderboard_rejects_when_secret_not_configured(monkeypatch):
    monkeypatch.delenv("LEADERBOARD_SHARED_SECRET", raising=False)
    response = client.post(
        "/leaderboard/",
        json={
            "model_name": "U-Shape",
            "dataset_name": "UIEB",
            "metric_name": "ssim",
            "score": 0.9,
        },
        headers={"X-Leaderboard-Secret": ""},
    )
    assert response.status_code == 401


def test_leaderboard_accepts_correct_secret(monkeypatch):
    monkeypatch.setenv("LEADERBOARD_SHARED_SECRET", "top-secret")
    recorded = []
    monkeypatch.setattr(
        supabase_api,
        "insert_leaderboard_entry",
        lambda model_name, dataset_name, metric_name, score: recorded.append(
            (model_name, dataset_name, metric_name, score)
        )
        or True,
    )

    response = client.post(
        "/leaderboard/",
        json={
            "model_name": "U-Shape",
            "dataset_name": "UIEB",
            "metric_name": "ssim",
            "score": 0.9,
        },
        headers={"X-Leaderboard-Secret": "top-secret"},
    )

    assert response.status_code == 201
    assert recorded == [("U-Shape", "UIEB", "ssim", 0.9)]
