# src/prediction.py
# Loads the trained model and predicts the class of one image.

import io
import json
import os
import sys

import numpy as np
from PIL import Image

# Make "from model import ..." work no matter where the script is started from
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import build_model, IMG_SIZE

# Paths are built from this file's location, so they work from any folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_model.h5")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

model = None
class_names = []


def load_model():
    """Rebuild the model structure, then load the trained weights from the .h5 file."""
    global model, class_names
    with open(CLASSES_PATH) as f:
        class_names = json.load(f)
    model = build_model(num_classes=len(class_names))
    model.load_weights(MODEL_PATH)


def predict_image(image_bytes):
    """Takes the raw bytes of an image, returns (class name, confidence, all scores)."""
    if model is None:
        load_model()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)  # shape (1, 224, 224, 3)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    scores = {class_names[i]: float(p) for i, p in enumerate(probs)}
    return class_names[idx], float(probs[idx]), scores


if __name__ == "__main__":
    # Quick self-test: only runs when you start this file directly
    load_model()
    print("Model loaded OK")
    print("Classes:", class_names)