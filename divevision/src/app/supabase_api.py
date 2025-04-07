from io import BytesIO
import os
from pathlib import Path
import uuid
import storage3
import supabase
from dotenv import load_dotenv
import logging
from PIL import Image

logger = logging.getLogger(__name__)

assert load_dotenv(Path(".env").resolve())
url: str = os.environ["SUPABASE_URL"]
key: str = os.environ["SUPABASE_KEY"]
supabase_client = supabase.create_client(url, key)


def create_user(
    mail: str,
    password: str,
) -> bool:
    try:
        supabase_client.auth.sign_up(
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
) -> bool:
    try:
        supabase_client.auth.sign_in_with_password(
            {
                "email": mail,
                "password": password,
            }
        )
    except supabase.AuthApiError as e:
        logger.error(e)
        return False
    else:
        return True


def upload_image(
    filepath: str,
) -> bool:
    try:
        # Get current user
        response = supabase_client.auth.get_session()
        if response is None:
            raise supabase.AuthApiError(message="User is not logged in")

        # Upload image to Supabase
        supabase_client.storage.from_("images").upload(
            # Store the file to s3://<bucket>/<user_uuid>/<image_uuid>.jpg
            path=f"{response.user.id}/{uuid.uuid4()}.jpg",
            file=filepath,
            file_options={
                "content-type": "image/jpeg",
            },
        )
    except (storage3.exceptions.StorageApiError, supabase.AuthApiError) as e:
        logger.error(e)
        return False
    else:
        return True


def download_image(
    remotepath: str,
) -> Image.Image | None:
    try:
        # Get current user
        response = supabase_client.auth.get_session()
        if response is None:
            raise supabase.AuthApiError(message="User is not logged in")

        # Download image from Supabase
        response: bytes = supabase_client.storage.from_("images").download(
            remotepath,
        )
        # Create image from downloaded bytes
        image = Image.open(BytesIO(response))
    except (storage3.exceptions.StorageApiError, supabase.AuthApiError) as e:
        logger.error(e)
        return None
    else:
        return image


if __name__ == "__main__":
    create_user("test@example.com", "123456")
    sign_in_user("test@example.com", "123456")
    upload_image("divevision/data/LSUI/GT/0.jpg")
    download_image(
        "77683a88-9517-45ce-b7a9-fbe988d52e5e/295d5a62-cf0d-465d-bf2d-c78ab3377608.jpg"
    )
