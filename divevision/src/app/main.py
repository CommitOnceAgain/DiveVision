import io
import os
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel, ConfigDict

from divevision.src.app import supabase_api
from divevision.src.models.u_shape_model import UShapeModelWrapper

app = FastAPI()


class Credentials(BaseModel):
    email: str
    password: str


class AuthSession(BaseModel):
    access_token: str
    refresh_token: str
    user_id: str


class LeaderboardEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    dataset_name: str
    metric_name: str
    score: float


@app.get(
    "/",
    response_class=HTMLResponse,
)
async def root():
    return """
<body>
<form action="/image/" enctype="multipart/form-data" method="post">
<input name="file" type="file">
<input type="submit">
</form>
</body>
"""


@app.post("/signup/", status_code=201)
async def signup(credentials: Credentials):
    if not supabase_api.create_user(credentials.email, credentials.password):
        raise HTTPException(status_code=400, detail="Could not create user")
    return {"detail": "user created"}


@app.post("/login/", response_model=AuthSession)
async def login(credentials: Credentials):
    session = supabase_api.sign_in_user(credentials.email, credentials.password)
    if session is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return session


def auth_tokens(
    authorization: Annotated[str, Header()],
    x_refresh_token: Annotated[str, Header()],
) -> tuple[str, str]:
    """Extract the Supabase session tokens returned by /login/ from request headers."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return authorization.removeprefix("Bearer "), x_refresh_token


@app.post(
    "/image/",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def upload_file(
    file: UploadFile = File(...),
    tokens: tuple[str, str] = Depends(auth_tokens),
):
    """Run the default model on an uploaded photo and persist both copies.

    Keeps the original synchronous request/response shape (upload in,
    processed PNG bytes out) - the only change is that the original and
    processed images are now saved to storage and tracked in the `photos`
    table instead of being discarded after the response is sent.
    """
    access_token, refresh_token = tokens

    model = UShapeModelWrapper()
    contents = await file.read()
    with io.BytesIO(contents) as f:
        image = Image.open(f)
        output: Image.Image = model.predict(image)[0]  # predict() returns a list

    buffer = io.BytesIO()
    output.save(buffer, "PNG")
    processed_bytes = buffer.getvalue()

    original_path = supabase_api.upload_image(
        access_token, refresh_token, contents, supabase_api.IMAGES_BUCKET
    )
    if original_path is None:
        raise HTTPException(
            status_code=502, detail="Could not store the original photo"
        )

    photo_id = supabase_api.create_photo(
        access_token, refresh_token, original_path, model.name
    )
    if photo_id is None:
        raise HTTPException(status_code=502, detail="Could not record the photo")

    # Store the processed result under the same relative path as the original.
    processed_path = supabase_api.upload_image(
        access_token,
        refresh_token,
        processed_bytes,
        supabase_api.PROCESSED_IMAGES_BUCKET,
        path=original_path,
    )
    if processed_path is None:
        supabase_api.mark_photo_failed(access_token, refresh_token, photo_id)
        raise HTTPException(
            status_code=502, detail="Could not store the processed photo"
        )

    supabase_api.mark_photo_completed(
        access_token, refresh_token, photo_id, processed_path
    )

    return Response(content=processed_bytes, media_type="image/png")


@app.delete("/photos/{photo_id}/", status_code=204)
async def delete_photo(
    photo_id: str,
    tokens: tuple[str, str] = Depends(auth_tokens),
):
    access_token, refresh_token = tokens
    if not supabase_api.delete_photo(access_token, refresh_token, photo_id):
        raise HTTPException(status_code=404, detail="Photo not found")
    return Response(status_code=204)


@app.delete("/account/", status_code=204)
async def delete_account(
    tokens: tuple[str, str] = Depends(auth_tokens),
):
    """Erase a user's account: every photo (storage + row), then the auth user itself."""
    access_token, refresh_token = tokens
    if not supabase_api.delete_account(access_token, refresh_token):
        raise HTTPException(status_code=400, detail="Could not delete account")
    return Response(status_code=204)


def leaderboard_auth(x_leaderboard_secret: Annotated[str, Header()]) -> None:
    """Shared-secret check for the internal leaderboard endpoint.

    Not a user-auth flow: this is called by a local MLflow benchmark script
    that never receives any Supabase key. Fails closed if the secret isn't
    configured server-side.
    """
    expected = os.environ.get("LEADERBOARD_SHARED_SECRET", "")
    if not expected or not secrets.compare_digest(x_leaderboard_secret, expected):
        raise HTTPException(status_code=401, detail="Invalid leaderboard secret")


@app.post("/leaderboard/", status_code=201)
async def record_leaderboard_entry(
    entry: LeaderboardEntry,
    _: None = Depends(leaderboard_auth),
):
    if not supabase_api.insert_leaderboard_entry(
        entry.model_name, entry.dataset_name, entry.metric_name, entry.score
    ):
        raise HTTPException(
            status_code=502, detail="Could not record leaderboard entry"
        )
    return {"detail": "recorded"}
