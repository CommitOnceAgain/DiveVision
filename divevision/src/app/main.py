from fastapi import FastAPI, UploadFile

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome!"}


@app.post("/images/")
async def upload_file(files: list[UploadFile]):
    return {"filenames": [file.filename for file in files]}
