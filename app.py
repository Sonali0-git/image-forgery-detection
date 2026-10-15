
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import numpy as np
import io
import os

app = FastAPI(title="Image Forgery Detection")

templates = Jinja2Templates(directory="templates")

MODEL_PATH = os.path.join(
    "model", "image_forgery_detection_final.keras"
)

THRESHOLD = 0.75
model = None


@app.on_event("startup")
def load_model():
    global model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"error": "Please upload a valid image."}

    try:
        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            return {"error": "Image must be smaller than 10 MB."}

        with Image.open(io.BytesIO(contents)) as image:
            image = image.convert("RGB").resize((224, 224))
            array = np.asarray(image, dtype=np.float32) / 255.0

        array = np.expand_dims(array, axis=0)

        probability = float(
            model.predict(array, verbose=0).reshape(-1)[0]
        )

        prediction = (
            "FORGED" if probability >= THRESHOLD else "REAL"
        )

        return {
            "prediction": prediction,
            "forged_probability": round(probability * 100, 2),
            "threshold": THRESHOLD
        }

    except (UnidentifiedImageError, OSError):
        return {"error": "Could not read this image file."}
