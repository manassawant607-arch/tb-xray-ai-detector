"""Train a tiny CNN on SYNTHETIC image data to produce tb_detection_model.h5.

This exists so the app's real-inference code path can run end-to-end (model
load -> preprocess -> model.predict -> threshold -> interpret) without the real
~4GB Kaggle TB Chest Radiography dataset and a multi-hour GPU training run.

The model is trained on class-correlated synthetic gradients (lighter ~ normal,
darker ~ tb). It is NOT medically meaningful and must NOT be used for diagnosis.
For a real model, run ai_drp.ipynb with the real dataset.
"""
import os

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential

IMG_SIZE = (224, 224)
N_PER_CLASS = 150
OUT_DIR = "_synthetic_data"
MODEL_PATH = "tb_detection_model.h5"
EPOCHS = 6
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


def make_image(base_intensity, idx):
    """Synthetic chest-x-ray-like image: a class-correlated brightness gradient
    with a faint central blob and per-image noise."""
    w, h = IMG_SIZE
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w // 2, h // 2
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (w * 0.18) ** 2)))
    img = np.full((h, w), base_intensity, dtype=np.float32)
    img += blob * 30  # faint central "lung field"
    img += np.random.normal(0, 12, (h, w))
    img += (idx % 7) * 2  # small per-batch drift
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="L")


def build_dataset():
    os.makedirs(OUT_DIR, exist_ok=True)
    for cls, base in [("Normal", 200), ("Tuberculosis", 70)]:
        d = os.path.join(OUT_DIR, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(N_PER_CLASS):
            make_image(base, i).save(os.path.join(d, f"{cls}_{i:03d}.png"))
    print(f"wrote {N_PER_CLASS * 2} synthetic images to {OUT_DIR}/")


def build_model():
    return Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])


def main():
    build_dataset()
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255, validation_split=0.2)
    train = datagen.flow_from_directory(
        OUT_DIR, target_size=IMG_SIZE, batch_size=16,
        class_mode="binary", subset="training")
    val = datagen.flow_from_directory(
        OUT_DIR, target_size=IMG_SIZE, batch_size=16,
        class_mode="binary", subset="validation")
    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train, validation_data=val, epochs=EPOCHS, verbose=2)
    loss, acc = model.evaluate(val, verbose=0)
    print(f"val_loss={loss:.4f} val_acc={acc:.4f}")
    model.save(MODEL_PATH)
    print(f"saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
