# src/retrain.py
# Retrains the saved model using the training images + any newly uploaded images.

import json
import os
import shutil
import time
import uuid

import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

from model import build_model, IMG_SIZE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "test")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "waste_model.h5")
PREVIOUS_PATH = os.path.join(MODELS_DIR, "waste_model_previous.h5")
CLASSES_PATH = os.path.join(MODELS_DIR, "class_names.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

BATCH_SIZE = 32
SEED = 42
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def load_classes():
    with open(CLASSES_PATH) as f:
        return json.load(f)


def count_images(folder):
    """Number of image files in one folder (0 if the folder does not exist)."""
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS)])


def count_uploads():
    """How many new images are waiting in data/uploads, per class."""
    classes = load_classes()
    return {c: count_images(os.path.join(UPLOAD_DIR, c)) for c in classes}


def retrain(epochs=5, learning_rate=0.0001):
    classes = load_classes()
    start = time.time()

    # Make sure the upload folders exist (one per class)
    for c in classes:
        os.makedirs(os.path.join(UPLOAD_DIR, c), exist_ok=True)

    # ---- 1. Data processing: load + resize images to 224x224 ----
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, class_names=classes, validation_split=0.2, subset="training",
        seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, class_names=classes, validation_split=0.2, subset="validation",
        seed=SEED, image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, class_names=classes, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        label_mode="categorical", shuffle=False)

    uploads = count_uploads()
    n_new = sum(uploads.values())
    if n_new > 0:
        upload_ds = tf.keras.utils.image_dataset_from_directory(
            UPLOAD_DIR, class_names=classes, image_size=IMG_SIZE,
            batch_size=BATCH_SIZE, label_mode="categorical", seed=SEED)
        train_ds = train_ds.concatenate(upload_ds)   # old + new images together
    print(f"New uploaded images used for retraining: {n_new}")

    # Class weights from the counts of old + new images
    labels = []
    for i, c in enumerate(classes):
        labels += [i] * (count_images(os.path.join(TRAIN_DIR, c)) + uploads[c])
    weights = compute_class_weight(
        "balanced", classes=np.arange(len(classes)), y=np.array(labels))
    class_weights = dict(enumerate(weights))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    # ---- 2. Load the saved model as the starting point (pre-trained) ----
    model = build_model(num_classes=len(classes), learning_rate=learning_rate)
    model.load_weights(MODEL_PATH)

    # ---- 3. Retrain (early stopping + class weights, as in the notebook) ----
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True)
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        class_weight=class_weights, callbacks=[early_stop])

    # ---- 4. Test on the test set: accuracy, loss, precision, recall, F1 ----
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    y_true = np.concatenate([np.argmax(y.numpy(), axis=1) for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)

    metrics = {
        "accuracy": round(float(test_acc), 4),
        "loss": round(float(test_loss), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "new_images_used": n_new,
        "epochs_run": len(history.history["loss"]),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - start, 1),
    }

    # ---- 5. Save: keep a backup of the old model, then save the new one ----
    if os.path.exists(MODEL_PATH):
        shutil.copy(MODEL_PATH, PREVIOUS_PATH)
    model.save(MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # The new images now become part of the training data
    for c in classes:
        src_folder = os.path.join(UPLOAD_DIR, c)
        for name in os.listdir(src_folder):
            if name.lower().endswith(IMAGE_EXTS):
                new_name = f"upload_{uuid.uuid4().hex[:8]}_{name}"
                shutil.move(os.path.join(src_folder, name),
                            os.path.join(TRAIN_DIR, c, new_name))

    print("Retraining finished:", metrics)
    return metrics


if __name__ == "__main__":
    # Quick self-test: 2 short epochs (no uploads yet)
    retrain(epochs=2)