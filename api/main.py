import base64
import io
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DINO_CHECKPOINT = PROJECT_ROOT / "model/fish_disease_dino.pt"
SEVERITY_MODEL_PATH = PROJECT_ROOT / "model/severity.json"
TREATMENT_PATH = PROJECT_ROOT / "treatment_data.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def run_dino_worker(contents: bytes) -> dict:
    request = {"image": base64.b64encode(contents).decode("ascii")}
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "dino_worker.py")],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=180,
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        raise RuntimeError("DINOv2 worker failed to process the image.")
    return json.loads(result.stdout)


def severity_features(image: Image.Image) -> np.ndarray:
    image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    body_mask = np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255
    white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    red_low = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_high = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    lesion_mask = cv2.bitwise_or(white, cv2.bitwise_or(red_low, red_high))
    lesion_mask = cv2.bitwise_and(lesion_mask, lesion_mask, mask=body_mask)
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pixels = hsv[body_mask.astype(bool)]
    body_area = float(np.count_nonzero(body_mask))
    return np.array([[
        np.count_nonzero(lesion_mask) / body_area * 100,
        len(contours),
        max((cv2.contourArea(contour) for contour in contours), default=0),
        pixels[:, 0].mean(), pixels[:, 1].mean(), pixels[:, 2].mean(), body_area,
    ]], dtype=np.float32)


def choose_treatment(disease: str, severity: str, treatment_data: dict) -> tuple[dict, list[dict]]:
    options = treatment_data.get(disease, {}).get(severity, [])
    if not options:
        return {"text": "Consult an aquatic veterinarian.", "efficacy": 0.0, "environmental_impact": 0.0}, []
    ranked = sorted(options, key=lambda option: option["efficacy"] - 0.4 * option["environmental_impact"], reverse=True)
    return ranked[0], ranked


def image_base64(image: Image.Image) -> str:
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return base64.b64encode(output.getvalue()).decode("ascii")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DINO_CHECKPOINT.exists():
        raise RuntimeError("Run train_dino.py before starting the API.")
    checkpoint = json.loads("{}")
    app.state.device = "isolated worker"
    app.state.class_names = []
    app.state.severity_model = None
    if SEVERITY_MODEL_PATH.exists():
        app.state.severity_model = xgb.XGBClassifier()
        app.state.severity_model.load_model(SEVERITY_MODEL_PATH)
    app.state.treatment_data = json.loads(TREATMENT_PATH.read_text(encoding="utf-8"))
    yield


app = FastAPI(title="Fish Health Lens API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health(request: Request) -> dict:
    return {
        "status": "ok",
        "model": "DINOv2",
        "device": request.app.state.device,
        "severity": bool(request.app.state.severity_model),
        "treatments": len(request.app.state.treatment_data),
    }


@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)) -> dict:
    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 10 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image.load()
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from error

    state = request.app.state
    try:
        prediction = run_dino_worker(contents)
    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    disease = prediction["disease"]
    severity = None
    treatment = None
    treatment_options = []
    if state.severity_model is not None:
        severity_index = int(state.severity_model.predict(severity_features(image))[0])
        severity = ["mild", "moderate", "severe"][severity_index]
        treatment, treatment_options = choose_treatment(disease, severity, state.treatment_data)

    return {
        "disease": disease,
        "confidence": prediction["confidence"],
        "scores": prediction["scores"],
        "severity": severity,
        "severity_is_experimental": severity is not None,
        "treatment": treatment,
        "treatment_options": treatment_options,
        "treatment_notice": "Reference information only; confirm with an aquatic veterinarian.",
        "image_jpeg_base64": image_base64(image),
    }
