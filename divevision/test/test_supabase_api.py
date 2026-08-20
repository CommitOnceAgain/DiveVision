from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import storage3
import supabase
from PIL import Image

from divevision.src.app import supabase_api


def _postgrest_error(message="boom"):
    return supabase.PostgrestAPIError(
        {"message": message, "code": "500", "hint": None, "details": None}
    )


def _storage_error(message="boom"):
    return storage3.exceptions.StorageApiError(message=message, code="500", status=500)


@pytest.fixture
def mock_client(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(supabase_api, "get_client", lambda: client)
    return client


@pytest.fixture
def mock_admin_client(monkeypatch):
    client = MagicMock()
    monkeypatch.setattr(supabase_api, "get_admin_client", lambda: client)
    return client


def _session(user_id="user-123"):
    return SimpleNamespace(user=SimpleNamespace(id=user_id))


def _png_bytes():
    buffer = BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, "PNG")
    return buffer.getvalue()


# -- create_user / sign_in_user -----------------------------------------------


def test_create_user_success(mock_client):
    assert supabase_api.create_user("test@example.com", "123456") is True
    mock_client.auth.sign_up.assert_called_once_with(
        {"email": "test@example.com", "password": "123456"}
    )


def test_create_user_failure(mock_client):
    mock_client.auth.sign_up.side_effect = supabase.AuthApiError(
        message="already registered", status=400, code=None
    )
    assert supabase_api.create_user("test@example.com", "123456") is False


def test_sign_in_user_success(mock_client):
    mock_client.auth.sign_in_with_password.return_value = SimpleNamespace(
        session=SimpleNamespace(access_token="access", refresh_token="refresh"),
        user=SimpleNamespace(id="user-123"),
    )

    session = supabase_api.sign_in_user("test@example.com", "123456")

    assert session == {
        "access_token": "access",
        "refresh_token": "refresh",
        "user_id": "user-123",
    }


def test_sign_in_user_invalid_credentials(mock_client):
    mock_client.auth.sign_in_with_password.side_effect = supabase.AuthApiError(
        message="invalid credentials", status=400, code=None
    )
    assert supabase_api.sign_in_user("test@example.com", "wrong") is None


# -- upload_image / download_image / delete_image ------------------------------


def test_upload_image_generates_path_under_bucket(mock_client):
    mock_client.auth.get_session.return_value = _session()

    photo_id = supabase_api.upload_image(
        "access", "refresh", b"raw-bytes", supabase_api.IMAGES_BUCKET
    )

    mock_client.auth.set_session.assert_called_once_with("access", "refresh")
    assert photo_id.startswith("user-123/")
    assert photo_id.endswith(".jpg")
    mock_client.storage.from_.assert_called_with(supabase_api.IMAGES_BUCKET)
    upload_call = mock_client.storage.from_.return_value.upload
    assert upload_call.call_args.kwargs["path"] == photo_id
    assert upload_call.call_args.kwargs["file"] == b"raw-bytes"


def test_upload_image_sets_content_type_from_actual_image_bytes(mock_client):
    mock_client.auth.get_session.return_value = _session()

    supabase_api.upload_image(
        "access", "refresh", _png_bytes(), supabase_api.IMAGES_BUCKET
    )

    upload_call = mock_client.storage.from_.return_value.upload
    assert upload_call.call_args.kwargs["file_options"]["content-type"] == "image/png"


def test_upload_image_uses_explicit_path(mock_client):
    mock_client.auth.get_session.return_value = _session()

    path = supabase_api.upload_image(
        "access",
        "refresh",
        b"raw-bytes",
        supabase_api.PROCESSED_IMAGES_BUCKET,
        path="user-123/fixed.jpg",
    )

    assert path == "user-123/fixed.jpg"
    mock_client.storage.from_.assert_called_with(supabase_api.PROCESSED_IMAGES_BUCKET)


def test_upload_image_not_logged_in(mock_client):
    mock_client.auth.get_session.return_value = None
    assert (
        supabase_api.upload_image(
            "access", "refresh", b"raw-bytes", supabase_api.IMAGES_BUCKET
        )
        is None
    )


def test_upload_image_storage_error(mock_client):
    mock_client.auth.get_session.return_value = _session()
    mock_client.storage.from_.return_value.upload.side_effect = _storage_error()
    assert (
        supabase_api.upload_image(
            "access", "refresh", b"raw-bytes", supabase_api.IMAGES_BUCKET
        )
        is None
    )


def test_download_image_success(mock_client):
    mock_client.auth.get_session.return_value = _session()
    mock_client.storage.from_.return_value.download.return_value = _png_bytes()

    image = supabase_api.download_image(
        "access", "refresh", "user-123/photo.jpg", supabase_api.PROCESSED_IMAGES_BUCKET
    )

    assert image is not None
    mock_client.storage.from_.assert_called_with(supabase_api.PROCESSED_IMAGES_BUCKET)


def test_download_image_storage_error(mock_client):
    mock_client.auth.get_session.return_value = _session()
    mock_client.storage.from_.return_value.download.side_effect = _storage_error()
    assert (
        supabase_api.download_image(
            "access", "refresh", "user-123/photo.jpg", supabase_api.IMAGES_BUCKET
        )
        is None
    )


def test_delete_image_success(mock_client):
    mock_client.auth.get_session.return_value = _session()
    assert (
        supabase_api.delete_image(
            "access", "refresh", "user-123/photo.jpg", supabase_api.IMAGES_BUCKET
        )
        is True
    )
    mock_client.storage.from_.return_value.remove.assert_called_once_with(
        ["user-123/photo.jpg"]
    )


def test_delete_image_storage_error(mock_client):
    mock_client.auth.get_session.return_value = _session()
    mock_client.storage.from_.return_value.remove.side_effect = _storage_error()
    assert (
        supabase_api.delete_image(
            "access", "refresh", "user-123/photo.jpg", supabase_api.IMAGES_BUCKET
        )
        is False
    )


# -- photos table --------------------------------------------------------------


def test_create_photo_success(mock_client):
    mock_client.auth.get_session.return_value = _session()
    mock_client.table.return_value.insert.return_value.execute.return_value = (
        SimpleNamespace(data=[{"id": "photo-1"}])
    )

    photo_id = supabase_api.create_photo(
        "access", "refresh", "user-123/photo.jpg", "U-Shape"
    )

    assert photo_id == "photo-1"
    mock_client.table.assert_called_with("photos")
    insert_call = mock_client.table.return_value.insert
    insert_call.assert_called_once_with(
        {
            "user_id": "user-123",
            "original_path": "user-123/photo.jpg",
            "model_name": "U-Shape",
        }
    )


def test_create_photo_failure(mock_client):
    mock_client.auth.get_session.return_value = _session()
    mock_client.table.return_value.insert.return_value.execute.side_effect = (
        _postgrest_error()
    )
    assert (
        supabase_api.create_photo("access", "refresh", "user-123/photo.jpg", "U-Shape")
        is None
    )


def test_mark_photo_completed(mock_client):
    assert (
        supabase_api.mark_photo_completed(
            "access", "refresh", "photo-1", "user-123/photo.jpg"
        )
        is True
    )
    update_call = mock_client.table.return_value.update
    update_call.assert_called_once_with(
        {"status": "completed", "processed_path": "user-123/photo.jpg"}
    )


def test_mark_photo_failed(mock_client):
    assert supabase_api.mark_photo_failed("access", "refresh", "photo-1") is True
    mock_client.table.return_value.update.assert_called_once_with({"status": "failed"})


def test_get_photo_found(mock_client):
    mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = SimpleNamespace(
        data=[{"id": "photo-1", "original_path": "user-123/photo.jpg"}]
    )
    photo = supabase_api.get_photo("access", "refresh", "photo-1")
    assert photo == {"id": "photo-1", "original_path": "user-123/photo.jpg"}


def test_get_photo_not_found(mock_client):
    mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = SimpleNamespace(
        data=[]
    )
    assert supabase_api.get_photo("access", "refresh", "photo-1") is None


def test_delete_photo_removes_objects_and_row(mock_client, monkeypatch):
    monkeypatch.setattr(
        supabase_api,
        "get_photo",
        lambda access_token, refresh_token, photo_id: {
            "id": photo_id,
            "original_path": "user-123/photo.jpg",
            "processed_path": "user-123/photo-processed.jpg",
        },
    )
    deleted = []
    monkeypatch.setattr(
        supabase_api,
        "delete_image",
        lambda access_token, refresh_token, path, bucket: deleted.append((path, bucket))
        or True,
    )

    assert supabase_api.delete_photo("access", "refresh", "photo-1") is True

    assert ("user-123/photo.jpg", supabase_api.IMAGES_BUCKET) in deleted
    assert (
        "user-123/photo-processed.jpg",
        supabase_api.PROCESSED_IMAGES_BUCKET,
    ) in deleted
    mock_client.table.return_value.delete.return_value.eq.assert_called_once_with(
        "id", "photo-1"
    )


def test_delete_photo_not_found(mock_client, monkeypatch):
    monkeypatch.setattr(
        supabase_api,
        "get_photo",
        lambda access_token, refresh_token, photo_id: None,
    )
    assert supabase_api.delete_photo("access", "refresh", "photo-1") is False


# -- delete_account -------------------------------------------------------------


def test_delete_account_removes_photos_and_auth_user(
    mock_client, mock_admin_client, monkeypatch
):
    mock_client.auth.get_session.return_value = _session()
    mock_client.table.return_value.select.return_value.execute.return_value = (
        SimpleNamespace(
            data=[
                {
                    "original_path": "user-123/a.jpg",
                    "processed_path": "user-123/a-processed.jpg",
                },
                {"original_path": "user-123/b.jpg", "processed_path": None},
            ]
        )
    )
    deleted = []
    monkeypatch.setattr(
        supabase_api,
        "delete_image",
        lambda access_token, refresh_token, path, bucket: deleted.append((path, bucket))
        or True,
    )

    assert supabase_api.delete_account("access", "refresh") is True

    assert ("user-123/a.jpg", supabase_api.IMAGES_BUCKET) in deleted
    assert ("user-123/a-processed.jpg", supabase_api.PROCESSED_IMAGES_BUCKET) in deleted
    assert ("user-123/b.jpg", supabase_api.IMAGES_BUCKET) in deleted
    assert len(deleted) == 3  # b.jpg has no processed_path, so nothing extra deleted
    mock_admin_client.auth.admin.delete_user.assert_called_once_with("user-123")


def test_delete_account_not_logged_in(mock_client, mock_admin_client):
    mock_client.auth.get_session.return_value = None
    assert supabase_api.delete_account("access", "refresh") is False
    mock_admin_client.auth.admin.delete_user.assert_not_called()


def test_delete_account_admin_delete_failure(mock_client, mock_admin_client):
    mock_client.auth.get_session.return_value = _session()
    mock_client.table.return_value.select.return_value.execute.return_value = (
        SimpleNamespace(data=[])
    )
    mock_admin_client.auth.admin.delete_user.side_effect = supabase.AuthApiError(
        message="boom", status=500, code=None
    )
    assert supabase_api.delete_account("access", "refresh") is False


# -- leaderboard ------------------------------------------------------------------


def test_insert_leaderboard_entry_success(mock_admin_client):
    assert supabase_api.insert_leaderboard_entry("U-Shape", "UIEB", "ssim", 0.9) is True
    mock_admin_client.table.assert_called_with("leaderboard")
    mock_admin_client.table.return_value.insert.assert_called_once_with(
        {
            "model_name": "U-Shape",
            "dataset_name": "UIEB",
            "metric_name": "ssim",
            "score": 0.9,
        }
    )


def test_insert_leaderboard_entry_failure(mock_admin_client):
    mock_admin_client.table.return_value.insert.return_value.execute.side_effect = (
        _postgrest_error()
    )
    assert (
        supabase_api.insert_leaderboard_entry("U-Shape", "UIEB", "ssim", 0.9) is False
    )
