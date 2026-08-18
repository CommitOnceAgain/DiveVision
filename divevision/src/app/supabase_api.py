from io import BytesIO
import logging
import os
import uuid

import storage3
import supabase
from dotenv import load_dotenv
from PIL import Image

logger = logging.getLogger(__name__)

load_dotenv()

SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")
SUPABASE_SERVICE_ROLE_KEY: str = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

IMAGES_BUCKET = "images"
PROCESSED_IMAGES_BUCKET = "processedimages"

_STORAGE_ERRORS = (storage3.exceptions.StorageApiError, supabase.AuthApiError)
_POSTGREST_ERRORS = (supabase.PostgrestAPIError, supabase.AuthApiError)


def get_client() -> supabase.Client:
    """Build a fresh, unauthenticated Supabase client using the public anon key.

    A new client is created per call (rather than sharing one module-level
    client) so that signing in as one user never leaks that session into a
    concurrent request for another user.
    """
    return supabase.create_client(SUPABASE_URL, SUPABASE_KEY)


def get_admin_client() -> supabase.Client:
    """Build a client authenticated with the service-role key.

    Only used server-side for operations that have no non-admin equivalent
    (deleting an auth user, writing leaderboard rows outside any user
    session). This key must never be handed to a caller outside the backend.
    """
    return supabase.create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def _client_for_session(access_token: str, refresh_token: str) -> supabase.Client:
    client = get_client()
    client.auth.set_session(access_token, refresh_token)
    return client


def _not_logged_in_error() -> supabase.AuthApiError:
    return supabase.AuthApiError(message="User is not logged in", status=401, code=None)


def create_user(
    mail: str,
    password: str,
) -> bool:
    try:
        get_client().auth.sign_up(
            {
                "email": mail,
                "password": password,
            }
        )
    except (supabase.AuthApiError, supabase.AuthInvalidCredentialsError) as e:
        logger.error(e)
        return False
    else:
        return True


def sign_in_user(
    mail: str,
    password: str,
) -> dict | None:
    """Sign in and return the session tokens the caller needs for later calls.

    Returns a dict with `access_token`, `refresh_token` and `user_id`, or
    None if authentication failed.
    """
    try:
        response = get_client().auth.sign_in_with_password(
            {
                "email": mail,
                "password": password,
            }
        )
    except supabase.AuthApiError as e:
        logger.error(e)
        return None

    if response.session is None or response.user is None:
        return None

    return {
        "access_token": response.session.access_token,
        "refresh_token": response.session.refresh_token,
        "user_id": response.user.id,
    }


def upload_image(
    access_token: str,
    refresh_token: str,
    file: bytes,
    bucket: str,
    path: str | None = None,
) -> str | None:
    """Upload image bytes as the signed-in user, returning the remote storage path.

    If `path` is omitted, a fresh `<user_id>/<uuid>.jpg` path is generated;
    pass it explicitly to store the processed result under the same path as
    the original it was derived from.
    """
    try:
        client = _client_for_session(access_token, refresh_token)

        session = client.auth.get_session()
        if session is None:
            raise _not_logged_in_error()

        remote_path = path or f"{session.user.id}/{uuid.uuid4()}.jpg"
        client.storage.from_(bucket).upload(
            path=remote_path,
            file=file,
            file_options={
                "content-type": "image/jpeg",
            },
        )
    except _STORAGE_ERRORS as e:
        logger.error(e)
        return None
    else:
        return remote_path


def download_image(
    access_token: str,
    refresh_token: str,
    remotepath: str,
    bucket: str,
) -> Image.Image | None:
    """Download an image as the signed-in user."""
    try:
        client = _client_for_session(access_token, refresh_token)

        session = client.auth.get_session()
        if session is None:
            raise _not_logged_in_error()

        response: bytes = client.storage.from_(bucket).download(remotepath)
        image = Image.open(BytesIO(response))
    except _STORAGE_ERRORS as e:
        logger.error(e)
        return None
    else:
        return image


def delete_image(
    access_token: str,
    refresh_token: str,
    remotepath: str,
    bucket: str,
) -> bool:
    try:
        client = _client_for_session(access_token, refresh_token)

        session = client.auth.get_session()
        if session is None:
            raise _not_logged_in_error()

        client.storage.from_(bucket).remove([remotepath])
    except _STORAGE_ERRORS as e:
        logger.error(e)
        return False
    else:
        return True


def create_photo(
    access_token: str,
    refresh_token: str,
    original_path: str,
    model_name: str,
) -> str | None:
    """Insert a `photos` row (status "processing") and return its id."""
    try:
        client = _client_for_session(access_token, refresh_token)

        session = client.auth.get_session()
        if session is None:
            raise _not_logged_in_error()

        response = (
            client.table("photos")
            .insert(
                {
                    "user_id": session.user.id,
                    "original_path": original_path,
                    "model_name": model_name,
                }
            )
            .execute()
        )
    except _POSTGREST_ERRORS as e:
        logger.error(e)
        return None
    else:
        return response.data[0]["id"]


def mark_photo_completed(
    access_token: str,
    refresh_token: str,
    photo_id: str,
    processed_path: str,
) -> bool:
    try:
        client = _client_for_session(access_token, refresh_token)
        client.table("photos").update(
            {"status": "completed", "processed_path": processed_path}
        ).eq("id", photo_id).execute()
    except _POSTGREST_ERRORS as e:
        logger.error(e)
        return False
    else:
        return True


def mark_photo_failed(
    access_token: str,
    refresh_token: str,
    photo_id: str,
) -> bool:
    try:
        client = _client_for_session(access_token, refresh_token)
        client.table("photos").update({"status": "failed"}).eq("id", photo_id).execute()
    except _POSTGREST_ERRORS as e:
        logger.error(e)
        return False
    else:
        return True


def get_photo(
    access_token: str,
    refresh_token: str,
    photo_id: str,
) -> dict | None:
    """Fetch a photo row. RLS scopes this to rows owned by the signed-in user."""
    try:
        client = _client_for_session(access_token, refresh_token)
        response = client.table("photos").select("*").eq("id", photo_id).execute()
    except _POSTGREST_ERRORS as e:
        logger.error(e)
        return None

    if not response.data:
        return None
    return response.data[0]


def delete_photo(
    access_token: str,
    refresh_token: str,
    photo_id: str,
) -> bool:
    """Delete one photo: both storage objects (if present) and its row."""
    photo = get_photo(access_token, refresh_token, photo_id)
    if photo is None:
        return False

    if photo["original_path"]:
        delete_image(access_token, refresh_token, photo["original_path"], IMAGES_BUCKET)
    if photo["processed_path"]:
        delete_image(
            access_token,
            refresh_token,
            photo["processed_path"],
            PROCESSED_IMAGES_BUCKET,
        )

    try:
        client = _client_for_session(access_token, refresh_token)
        client.table("photos").delete().eq("id", photo_id).execute()
    except _POSTGREST_ERRORS as e:
        logger.error(e)
        return False
    else:
        return True


def delete_account(
    access_token: str,
    refresh_token: str,
) -> bool:
    """Erase a user: every photo's storage objects, then the auth user itself.

    Deleting the auth user cascades (via the `photos.user_id` foreign key)
    to remove every remaining `photos` row, so rows do not need to be
    deleted one by one here - only the storage objects, which have no such
    cascade.
    """
    try:
        client = _client_for_session(access_token, refresh_token)

        session = client.auth.get_session()
        if session is None:
            raise _not_logged_in_error()
        user_id = session.user.id

        photos = (
            client.table("photos").select("original_path, processed_path").execute()
        )
    except _POSTGREST_ERRORS as e:
        logger.error(e)
        return False

    for photo in photos.data:
        if photo["original_path"]:
            delete_image(
                access_token, refresh_token, photo["original_path"], IMAGES_BUCKET
            )
        if photo["processed_path"]:
            delete_image(
                access_token,
                refresh_token,
                photo["processed_path"],
                PROCESSED_IMAGES_BUCKET,
            )

    try:
        get_admin_client().auth.admin.delete_user(user_id)
    except supabase.AuthApiError as e:
        logger.error(e)
        return False
    else:
        return True


def insert_leaderboard_entry(
    model_name: str,
    dataset_name: str,
    metric_name: str,
    score: float,
) -> bool:
    """Write a leaderboard row using the service-role key.

    Called from the `/leaderboard` endpoint on behalf of a local benchmark
    script, which never receives any Supabase key itself.
    """
    try:
        get_admin_client().table("leaderboard").insert(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "metric_name": metric_name,
                "score": score,
            }
        ).execute()
    except _POSTGREST_ERRORS as e:
        logger.error(e)
        return False
    else:
        return True
