"""Evaluate a trained TB model: metrics + confusion matrix + Grad-CAM.

Run after `python train_real_model.py` has produced `tb_detection_model.h5`.

  python evaluate_model.py                       # uses tb_detection_model.h5
  python evaluate_model.py --model tb_detector_ai.h5 --data TB_Chest_Radiography_Database
  python evaluate_model.py --gradcam samples/sample_tb_like.png

Outputs:
  - precision / recall / F1 / AUC printed to stdout
  - confusion_matrix.png saved to disk
  - gradcam_<input>.png heatmap overlay saved to disk (with --gradcam)

Not a medical device; for research/education only.
"""

import argparse
import os

import numpy as np

IMG_SIZE = (224, 224)
DEFAULT_MODEL = "tb_detection_model.h5"
DEFAULT_DATA = "TB_Chest_Radiography_Database"
THRESHOLD = 0.5


def find_data_dir(path):
    subs = {"Normal", "Tuberculosis"}
    for c in [path, os.path.join(path, "TB_Chest_Radiography_Database")]:
        if c and os.path.isdir(c) and subs.issubset(set(os.listdir(c))):
            return c
    return None


def load_val_generator(data_dir, seed=42):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    gen = ImageDataGenerator(
        rescale=1.0 / 255, validation_split=0.2).flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=32,
        class_mode="binary", subset="validation", seed=seed, shuffle=False)
    return gen


def collect_predictions(model, gen):
    """Return (y_true, y_prob) over the whole validation set."""
    probs = model.predict(gen, verbose=0).ravel()
    y_true = gen.classes
    return y_true, probs


def compute_metrics(y_true, y_prob, threshold=THRESHOLD):
    """precision/recall/F1/AUC + confusion-matrix components."""
    from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                                 roc_auc_score)
    y_pred = (y_prob > threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                  zero_division=0)
    try:
        roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold, "precision": float(p), "recall": float(r),
        "f1": float(f1), "roc_auc": roc,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def save_confusion_matrix(metrics, out_path="confusion_matrix.png"):
    """Render a 2x2 confusion matrix PNG (TB=positive class)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array([[metrics["tn"], metrics["fp"]],
                   [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "TB"])
    ax.set_yticklabels(["Normal", "TB"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved confusion matrix -> {out_path}")


def find_last_conv_layer(model):
    """Last Conv2D layer name (works for both the CNN and MobileNetV2 models)."""
    from tensorflow.keras.layers import Conv2D
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    raise ValueError("no Conv2D layer found in model")


def gradcam(model, image, last_conv_name=None):
    """Grad-CAM heatmap (HxW) for a single PIL image, normalized to [0,1]."""
    import tensorflow as tf
    from PIL import Image as PILImage
    img = PILImage.open(image).convert("RGB").resize(IMG_SIZE) if isinstance(image, str) \
        else image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    x = np.expand_dims(arr, axis=0)

    name = last_conv_name or find_last_conv_layer(model)
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(name).output, model.outputs[0]])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x)
        loss = preds[:, 0]
    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    mx = heatmap.max()
    heatmap = np.maximum(heatmap, 0) / (mx + 1e-8)
    return heatmap, float(preds[0][0])


def save_gradcam(model, image_path, out_path=None):
    """Overlay Grad-CAM heatmap on the X-ray and save PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    heatmap, pred = gradcam(model, image_path)
    base = PILImage.open(image_path).convert("RGB").resize(IMG_SIZE)
    out_path = out_path or f"gradcam_{os.path.basename(image_path)}"
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(base)
    ax.imshow(heatmap, cmap="jet", alpha=0.4,
              extent=(0, IMG_SIZE[0], IMG_SIZE[1], 0))
    ax.set_title(f"Grad-CAM  (pred={pred:.3f})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved grad-cam -> {out_path}  (pred={pred:.3f})")


def main():
    p = argparse.ArgumentParser(description="Evaluate a trained TB model.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="path to .h5/.keras model")
    p.add_argument("--data", default=DEFAULT_DATA, help="dataset root dir")
    p.add_argument("--gradcam", help="optional image path for a Grad-CAM heatmap")
    p.add_argument("--threshold", type=float, default=THRESHOLD)
    p.add_argument("--metrics", action="store_true",
                   help="always show precision/recall/f1/auc (shown by default)")
    args = p.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: model not found: {args.model}")
        print("Run `python train_real_model.py` first.")
        return 1

    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)
    print(f"loaded model: {args.model}")

    if args.gradcam:
        save_gradcam(model, args.gradcam)
        return 0

    data_dir = find_data_dir(args.data)
    if not data_dir:
        print(f"ERROR: no TB dataset found at '{args.data}'.")
        print("Cannot compute validation metrics without the dataset.")
        return 1

    gen = load_val_generator(data_dir)
    y_true, y_prob = collect_predictions(model, gen)
    m = compute_metrics(y_true, y_prob, threshold=args.threshold)
    print("\n=== Validation metrics ===")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    save_confusion_matrix(m)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
