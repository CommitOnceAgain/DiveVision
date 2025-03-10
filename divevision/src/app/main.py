import io
from fastapi import FastAPI, File, Response, UploadFile
from PIL import Image
from fastapi.responses import HTMLResponse
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
    model = UShapeModelWrapper()
    with file.file as f:
        image = Image.open(f)
        output: Image.Image = model.predict(image)[0]  # predict() returns a list

    # Convert the image as PNG instead of raw data before returning it
    buffer = io.BytesIO()
    output.save(buffer, "PNG")

    return Response(content=buffer.getvalue(), media_type="image/png")
