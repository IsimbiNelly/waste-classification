# src/api.py
# The API: web page, predict, uptime, bulk upload, retraining trigger, stats, metrics and database.

import json
import os
import time
import uuid

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import database
import prediction
import retrain as retrain_module

app = FastAPI(title="Waste Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()
AUTO_RETRAIN_THRESHOLD = 50   # retrain automatically when this many new images are waiting
IMAGE_EXTS = (".jpg", ".jpeg", ".png")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

request_count = {"predictions": 0}
retrain_status = {"state": "idle", "message": "No retraining yet", "last_metrics": None}


@app.on_event("startup")
def startup():
    """Create the database tables and load the model once when the API starts."""
    database.init_db()
    prediction.load_model()


# ---------- Web page ----------
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ---------- Uptime ----------
@app.get("/health")
def health():
    """Model up-time: is the API alive and how long has it been running?"""
    return {
        "status": "up",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "model_loaded": prediction.model is not None,
        "classes": prediction.class_names,
        "predictions_served": request_count["predictions"],
        "retrain_state": retrain_status["state"],
    }


# ---------- Prediction ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload one image, get the predicted waste class."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")
    image_bytes = await file.read()
    try:
        label, confidence, scores = prediction.predict_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this image.")
    request_count["predictions"] += 1
    return {"prediction": label, "confidence": confidence, "scores": scores}


# ---------- Bulk upload of new training data ----------
@app.post("/upload")
async def upload(background_tasks: BackgroundTasks,
                 label: str = Form(...),
                 files: list[UploadFile] = File(...)):
    """Upload many images of ONE class. They are saved to disk and logged in the database."""
    if label not in prediction.class_names:
        raise HTTPException(status_code=400,
                            detail=f"Label must be one of {prediction.class_names}")

    folder = os.path.join(retrain_module.UPLOAD_DIR, label)
    os.makedirs(folder, exist_ok=True)

    saved = 0
    for f in files:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in IMAGE_EXTS:
            continue    # skip files that are not images
        name = f"{uuid.uuid4().hex[:8]}{ext}"
        path = os.path.join(folder, name)
        with open(path, "wb") as out:
            out.write(await f.read())
        database.add_upload(name, label, path)    # save a record in the database
        saved += 1

    pending = sum(retrain_module.count_uploads().values())
    auto = False
    if pending >= AUTO_RETRAIN_THRESHOLD and retrain_status["state"] != "running":
        retrain_status.update(state="running", message="Automatic retraining started")
        background_tasks.add_task(run_retraining)
        auto = True

    return {"saved": saved, "label": label, "pending_images": pending,
            "auto_retrain_triggered": auto}


# ---------- Retraining ----------
def run_retraining():
    """Runs in the background so the API stays responsive."""
    try:
        metrics = retrain_module.retrain()
        database.log_retrain(metrics)       # save the scores of this run
        database.mark_uploads_used()        # the waiting uploads are now part of training
        prediction.load_model()             # use the new model right away
        retrain_status.update(state="done", message="Retraining finished",
                              last_metrics=metrics)
    except Exception as e:
        retrain_status.update(state="failed", message=f"Retraining failed: {e}")


@app.post("/retrain")
def retrain_button(background_tasks: BackgroundTasks):
    """The retrain button: starts retraining in the background."""
    if retrain_status["state"] == "running":
        raise HTTPException(status_code=409, detail="Retraining is already running.")
    retrain_status.update(state="running", message="Retraining started")
    background_tasks.add_task(run_retraining)
    return {"started": True, "pending_images": sum(retrain_module.count_uploads().values())}


@app.get("/retrain/status")
def retrain_progress():
    return retrain_status


# ---------- Database ----------
@app.get("/db/summary")
def db_summary():
    """What the database holds: logged uploads and the latest retraining runs."""
    return database.summary()


# ---------- Metrics and stats (for the web page) ----------
@app.get("/metrics")
def metrics():
    """Latest evaluation scores of the current model."""
    if not os.path.exists(retrain_module.METRICS_PATH):
        return {"available": False}
    with open(retrain_module.METRICS_PATH) as f:
        return {"available": True, **json.load(f)}


@app.get("/stats")
def stats():
    """Image counts per class: train, test and new uploads waiting."""
    classes = prediction.class_names
    pending = retrain_module.count_uploads()
    return {
        "classes": classes,
        "train": [retrain_module.count_images(os.path.join(retrain_module.TRAIN_DIR, c))
                  for c in classes],
        "test": [retrain_module.count_images(os.path.join(retrain_module.TEST_DIR, c))
                 for c in classes],
        "uploads_pending": [pending[c] for c in classes],
    }