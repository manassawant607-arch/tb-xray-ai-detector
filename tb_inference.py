"""Shared TB inference logic used by app.py and tests.

Keeps preprocessing, prediction interpretation, and model loading in one place
so the Gradio UI stays thin and the logic is unit-testable without a model.
"""

import os

import numpy as np

MODEL_PATH = os.environ.get("TB_MODEL_PATH", "tb_detection_model.h5")
IMG_SIZE = (224, 224)
THRESHOLD = 0.5

# Prototype labels (NOT clinically validated) used to demonstrate the pipeline.
TB_POSITIVE = ("TB Detected", "rpoB mutation detected",
               "Rifampicin Resistant (Possible MDR-TB)",
               "Bedaquiline + Linezolid + Levofloxacin")
TB_NEGATIVE = ("Normal", "No mutation detected", "Drug Sensitive",
               "Standard TB therapy")

_model = None


def preprocess(img):
    """Resize a PIL image to 224x224 RGB, normalize to [0,1], add batch dim."""
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def interpret_prediction(pred):
    """Map a sigmoid score in [0,1] to the (tb, mutation, resistance, treatment) tuple."""
    return TB_POSITIVE if pred > THRESHOLD else TB_NEGATIVE


def model_available(path=MODEL_PATH):
    """True if a trained Keras model file exists at `path`."""
    return os.path.exists(path)


def load_model(path=MODEL_PATH):
    """Lazily load and cache the Keras model. Requires tensorflow."""
    global _model
    if _model is None:
        import tensorflow as tf
        _model = tf.keras.models.load_model(path)
    return _model


def predict_tb(image, model):
    """Run real inference: preprocess -> model.predict -> interpret."""
    img = preprocess(image)
    pred = float(model.predict(img, verbose=0)[0][0])
    return interpret_prediction(pred)


def demo_predict(image):
    """Fallback used when no trained model is present.

    Returns the same 4-field layout so the UI is fully exercisable, but clearly
    labels every field as a demo placeholder. Uses image mean brightness only to
    make the output feel responsive, not as a real signal.
    """
    img = image.convert("RGB").resize(IMG_SIZE) if image is not None else None
    if img is None:
        brightness = 0.0
    else:
        brightness = float(np.array(img).mean() / 255.0)
    tag = "TB Detected" if brightness < 0.5 else "Normal"
    return (
        f"[DEMO] {tag} (no model loaded)",
        "[DEMO] mutation analysis unavailable",
        "[DEMO] drug-resistance prediction unavailable",
        "[DEMO] treatment recommendation unavailable",
    )
