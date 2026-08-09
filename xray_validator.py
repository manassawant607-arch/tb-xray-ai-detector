"""Chest X-ray validity gate — "right photo" (X-ray) vs "wrong photo".

Decides whether an uploaded image is a valid chest X-ray (right photo) or a
wrong photo (a colourful photograph, a screenshot, a blank or tiny image).
A wrong photo is rejected BEFORE detection so it is never misclassified as TB.

Two layers:
  1. Fast heuristic pre-filter: rejects obvious non-X-rays (too small, blank,
     flat, extreme aspect ratio) without loading any model.
  2. Trained RandomForest classifier (xray_gate_model.pkl) on 15 image
     statistics, trained on all 4,200 real X-rays + 510 non-X-ray negatives.
     This catches real-world photos that pass the simple heuristics (e.g. a
     photo of a cat) which a single saturation threshold cannot separate.

train_xray_gate.py trains the model; if the .pkl is absent the gate falls
back to the heuristic-only path so the app still runs.
"""

import os

import numpy as np

GATE_MODEL_PATH = "xray_gate_model.pkl"

# Heuristic pre-filter thresholds (conservative; only reject obvious junk).
MIN_DIM = 100            # reject postage-stamp images
MAX_ASPECT = 2.5         # reject very wide / tall strips
MIN_BRIGHTNESS = 0.04    # reject pure-black images
MAX_BRIGHTNESS = 0.985   # reject pure-white / blank images
MIN_STD = 0.015          # reject flat (uniform) images: X-rays have texture
MAX_SATURATION = 0.6     # reject high-saturation colour images (gradients, etc.)

_gate = None


def _to_rgb(image):
    if image is None:
        raise ValueError("no image provided")
    return image.convert("RGB")


def saturation(rgb_arr):
    """Mean per-pixel saturation in [0,1] on a normalized RGB array.

    0 = fully grey (no colour), 1 = fully saturated. Chest X-rays hover near 0.
    """
    r, g, b = rgb_arr[..., 0], rgb_arr[..., 1], rgb_arr[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    nonzero = mx > 1e-6
    sat = np.zeros_like(mx)
    sat[nonzero] = (mx[nonzero] - mn[nonzero]) / mx[nonzero]
    return float(sat.mean())


def extract_features(image):
    """15-dim vector of image statistics for the gate classifier."""
    img = _to_rgb(image)
    arr = np.asarray(img).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = arr.mean(axis=2)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    nonzero = mx > 1e-6
    sat = np.zeros_like(mx)
    sat[nonzero] = (mx[nonzero] - mn[nonzero]) / mx[nonzero]
    hist, _ = np.histogram(gray, bins=32, range=(0, 1))
    hist = hist / max(hist.sum(), 1)
    gy = np.abs(np.diff(gray, axis=0))
    gx = np.abs(np.diff(gray, axis=1))
    edge_density = float(gy.mean() + gx.mean())
    ch = max(gray.shape[0] // 6, 1)
    cw = max(gray.shape[1] // 6, 1)
    corner = np.concatenate([
        gray[:ch, :cw].ravel(), gray[:ch, -cw:].ravel(),
        gray[-ch:, :cw].ravel(), gray[-ch:, -cw:].ravel()])
    return np.array([
        float(sat.mean()),
        float(np.mean(mx - mn)),
        float((gray < 0.05).mean()),
        float((gray > 0.95).mean()),
        float(hist.max()),
        float(gray.mean()),
        float(gray.std()),
        edge_density,
        float(np.percentile(gray, 10)),
        float(np.percentile(gray, 90)),
        float(np.percentile(gray, 25)),
        float(np.percentile(gray, 75)),
        float(((gray > 0.2) & (gray < 0.8)).mean()),
        float(np.mean(np.var(arr, axis=2))),
        float(corner.mean()),
    ])


def _load_gate():
    """Lazily load the trained RandomForest gate (returns None if absent)."""
    global _gate
    if _gate is not None:
        return _gate
    if os.path.exists(GATE_MODEL_PATH):
        import pickle
        with open(GATE_MODEL_PATH, "rb") as fh:
            _gate = pickle.load(fh)
    return _gate


def is_chest_xray(image):
    """True if image looks like a valid chest X-ray (right photo), else False.

    Returns (bool_ok, reason_str). A wrong photo is rejected with a
    human-readable reason; a valid chest X-ray returns "valid chest X-ray".
    """
    try:
        img = _to_rgb(image)
    except (ValueError, OSError) as exc:
        return False, f"rejected: {exc}"

    w, h = img.size
    if w < MIN_DIM or h < MIN_DIM:
        return False, "rejected: image too small to be an X-ray"
    aspect = max(w, h) / min(w, h)
    if aspect > MAX_ASPECT:
        return False, "rejected: aspect ratio too extreme for a chest X-ray"

    arr = np.asarray(img).astype(np.float32) / 255.0
    gray = arr.mean(axis=2)
    mean = float(gray.mean())
    std = float(gray.std())
    if mean < MIN_BRIGHTNESS:
        return False, "rejected: image is too dark / near-black"
    if mean > MAX_BRIGHTNESS:
        return False, "rejected: image is too bright / near-blank"
    if std < MIN_STD:
        return False, "rejected: image is flat (no texture) — not an X-ray"
    if saturation(arr) > MAX_SATURATION:
        return False, "rejected: image is too colourful to be a chest X-ray"

    gate = _load_gate()
    if gate is None:
        return True, "valid chest X-ray (heuristic)"

    feats = extract_features(img).reshape(1, -1)
    scaled = gate["scaler"].transform(feats)
    proba = float(gate["model"].predict_proba(scaled)[0, 1])
    if proba < 0.5:
        return False, ("rejected: not a chest X-ray (looks like a photo, "
                       "screenshot, or other non-X-ray image)")
    return True, "valid chest X-ray"
