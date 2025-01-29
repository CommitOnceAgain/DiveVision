from fastapi import FastAPI, Response, UploadFile
from PIL import Image
from divevision.src.models.u_shape_model import UShapeModelWrapper

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome!"}


@app.post(
    "/image/",
    responses={200: {"content": {"image/png": {}}}},
    response_class=Response,
)
async def upload_file(file: UploadFile):
    model = UShapeModelWrapper()
    with file.file as f:
        image = Image.open(f)
        output: Image = model.predict(image)[0]  # predict() returns a list

    return Response(content=output.tobytes(), media_type="image/png")
