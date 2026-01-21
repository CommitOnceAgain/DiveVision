import io

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from divevision.src.models.u_shape_model import UShapeModelWrapper

app = FastAPI()


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


@app.post(
    "/image/",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def upload_file(file: UploadFile = File(...)):
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    # Read the file content
    file_content = await file.read()
    file_buffer = io.BytesIO(file_content)

    try:
        # Attempt to open the image
        image = Image.open(file_buffer)
        image.verify()  # Verify the image
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    else:
        model = UShapeModelWrapper()
        output: Image.Image = model.predict(image)[0]  # predict() returns a list

        # Convert the image as PNG instead of raw data before returning it
        buffer = io.BytesIO()
        output.save(buffer, "PNG")

        return Response(content=buffer.getvalue(), media_type="image/png")
