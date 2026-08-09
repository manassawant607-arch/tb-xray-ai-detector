"""Train the chest X-ray validity-gate classifier.

Trains a RandomForest classifier that distinguishes a real chest X-ray from
any other image (photo, screenshot, blank, tiny, colour pattern) using
hand-crafted image statistics. No deep learning is needed for the gate, so
it stays fast and dependency-light (numpy + scikit-learn).

Positive samples (chest X-ray): all images under
``TB_Chest_Radiography_Database/{Normal,Tuberculosis}/`` (4,200 real X-rays).
Negative samples (non-X-ray): real photos placed in ``neg_photos/`` plus a
large set of synthetically generated non-X-ray images (noise, gradients,
solid colours, random shapes, blobs, stripes).

The trained model + scaler is saved to ``xray_gate_model.pkl`` and loaded by
``xray_validator.is_chest_xray()``.

Usage:
    # after downloading the Kaggle dataset (python setup_kaggle.py)
    python train_xray_gate.py
    python train_xray_gate.py --neg-dir neg_photos --synth 500
"""

import argparse
import glob
import os
import pickle

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "xray_gate_model.pkl"
DEFAULT_DATA = "TB_Chest_Radiography_Database"
DEFAULT_NEG_DIR = "neg_photos"

FEATURE_NAMES = [
    "saturation", "channel_diff", "frac_black", "frac_white", "hist_peak",
    "brightness", "gray_std", "edge_density", "p10", "p90", "p25", "p75",
    "frac_mid", "channel_var", "corner_dark",
]


def extract_features(image):
    """Return a 15-dim vector of image statistics for the gate classifier."""
    img = image.convert("RGB")
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
    # corners tend to be dark (X-ray background outside the body)
    ch, cw = gray.shape[0] // 6, gray.shape[1] // 6
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
        float(((gray > 0.2) & (gray < 0.8)).mean()),  # mid-tone fraction
        float(np.mean(np.var(arr, axis=2))),  # per-pixel channel variance
        float(corner.mean()),  # corner darkness
    ])


def load_xrays(data_dir):
    """All real chest X-rays (positive class)."""
    paths = []
    for cls in ("Normal", "Tuberculosis"):
        paths += glob.glob(os.path.join(data_dir, cls, "*.png"))
        paths += glob.glob(os.path.join(data_dir, cls, "*.jpg"))
    return paths


def load_real_photos(neg_dir):
    """Real non-X-ray photos (negative class), if present."""
    paths = []
    if os.path.isdir(neg_dir):
        paths += glob.glob(os.path.join(neg_dir, "*.jpg"))
        paths += glob.glob(os.path.join(neg_dir, "*.jpeg"))
        paths += glob.glob(os.path.join(neg_dir, "*.png"))
    return [p for p in paths if os.path.getsize(p) > 100]


def synth_negatives(n, seed=42):
    """Yield `n` synthetic non-X-ray PIL images (diverse wrong photos)."""
    rng = np.random.RandomState(seed)
    w, h = 256, 256
    for _ in range(n):
        t = rng.randint(0, 7)
        if t == 0:  # colored noise
            a = rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
        elif t == 1:  # smooth gradient
            yy, xx = np.mgrid[0:h, 0:w]
            a = np.zeros((h, w, 3), np.uint8)
            for c in range(3):
                a[..., c] = (xx / w * rng.randint(0, 255)
                             + yy / h * rng.randint(0, 255)).clip(0, 255).astype(np.uint8)
        elif t == 2:  # solid colour
            a = np.full((h, w, 3), rng.randint(0, 255, 3), dtype=np.uint8)
        elif t == 3:  # random rectangles
            a = np.full((h, w, 3), rng.randint(0, 255), dtype=np.uint8)
            for _ in range(rng.randint(3, 10)):
                x0, y0 = rng.randint(0, w // 2), rng.randint(0, h // 2)
                bw, bh = rng.randint(20, w // 2), rng.randint(20, h // 2)
                a[y0:y0 + bh, x0:x0 + bw] = rng.randint(0, 255, 3)
        elif t == 4:  # gaussian blob
            cx, cy = rng.randint(0, w), rng.randint(0, h)
            yy, xx = np.mgrid[0:h, 0:w]
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                          / (2 * rng.randint(20, 60) ** 2))
            a = np.zeros((h, w, 3), np.uint8)
            for c in range(3):
                a[..., c] = (blob * rng.randint(50, 255)).clip(0, 255).astype(np.uint8)
        elif t == 5:  # striped pattern
            a = np.zeros((h, w, 3), np.uint8)
            for y in range(0, h, rng.randint(5, 30)):
                a[y:y + rng.randint(2, 10)] = rng.randint(0, 255, 3)
        else:  # blurred colour field (mimics a soft photo)
            a = rng.randint(0, 255, (h // 8, w // 8, 3)).astype(np.uint8)
            a = np.array(Image.fromarray(a).resize((w, h), Image.BILINEAR))
        yield Image.fromarray(a, "RGB")


def main():
    p = argparse.ArgumentParser(description="Train the X-ray validity-gate model.")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--neg-dir", default=DEFAULT_NEG_DIR)
    p.add_argument("--synth", type=int, default=500,
                   help="number of synthetic non-X-ray negatives to generate")
    args = p.parse_args()

    pos_paths = load_xrays(args.data)
    if not pos_paths:
        print(f"ERROR: no X-rays found under '{args.data}'.")
        print("Run `python setup_kaggle.py` first to download the dataset.")
        return 1

    print(f"positive (real X-rays): {len(pos_paths)}")
    feats = [extract_features(Image.open(f)) for f in pos_paths]
    Xpos = np.array(feats)

    neg_paths = load_real_photos(args.neg_dir)
    print(f"negative (real photos): {len(neg_paths)} from {args.neg_dir}/")
    Xneg_real = [extract_features(Image.open(f)) for f in neg_paths]
    Xneg_synth = [extract_features(img) for img in synth_negatives(args.synth)]
    Xneg = np.array(Xneg_real + Xneg_synth)
    print(f"negative (total incl. {args.synth} synthetic): {len(Xneg)}")

    X = np.vstack([Xpos, Xneg])
    y = np.concatenate([np.ones(len(Xpos)), np.zeros(len(Xneg))])

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=42)
    scores = cross_val_score(clf, Xs, y, cv=5, scoring="f1")
    print(f"CV F1: {scores.mean():.3f} +/- {scores.std():.3f}")
    clf.fit(Xs, y)
    print(f"train accuracy: {clf.score(Xs, y):.4f}")

    pred = clf.predict(Xs)
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    print(f"false positives (non-X-ray accepted): {fp}/{len(Xneg)}")
    print(f"false negatives (X-ray rejected): {fn}/{len(Xpos)}")

    bundle = {"model": clf, "scaler": scaler, "feature_names": FEATURE_NAMES}
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"saved -> {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
