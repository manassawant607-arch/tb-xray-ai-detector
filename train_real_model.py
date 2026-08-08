"""Train the TB detection model on REAL data locally — mirrors ai_drp.py.

ai_drp.py is a raw Colab export (uses `!kaggle`, `from google.colab import files`,
and has several duplicated `app.launch()` calls), so it cannot run as-is outside
Colab. This script reproduces the SAME real training pipeline from ai_drp.py but
runs locally:

  1. Load the TB Chest Radiography Database from a local directory
     (default: TB_Chest_Radiography_Database/{Normal,Tuberculosis}/).
  2. ImageDataGenerator rescale + 80/20 split at 224x224 (same as ai_drp.py).
  3. Train the from-scratch Sequential CNN -> save tb_detection_model.h5
     (identical architecture to ai_drp.py).
  4. Optional MobileNetV2 transfer-learning model -> tb_detector_ai.h5
     (same as ai_drp.py).
  5. Train a RandomForestRegressor on the TB trends CSV for the resistance
     prototype (same as ai_drp.py).

It uses the SAME label conventions as ai_drp.py / tb_inference.py.

Usage:
    # after downloading the Kaggle dataset into ./TB_Chest_Radiography_Database/
    python train_real_model.py
    python train_real_model.py --data path/to/dataset --epochs 10 --no-mobilenet
"""

import argparse
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DEFAULT_DATA = "TB_Chest_Radiography_Database"
DEFAULT_TRENDS = "Tuberculosis_Trends.csv"
CNN_MODEL_PATH = "tb_detection_model.h5"
TL_MODEL_PATH = "tb_detector_ai.h5"


def find_data_dir(path):
    """Resolve a dataset root that contains Normal/ and Tuberculosis/ subdirs.

    Accepts either the root containing those folders, or the root's parent.
    """
    subs = {"Normal", "Tuberculosis"}
    candidates = [path, os.path.join(path, "TB_Chest_Radiography_Database")]
    for c in candidates:
        if c and os.path.isdir(c) and subs.issubset(set(os.listdir(c))):
            return c
    return None


def make_generators(data_dir, seed=42):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)
    train = datagen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", subset="training", seed=seed)
    val = datagen.flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", subset="validation", seed=seed)
    print("class indices:", train.class_indices)
    return train, val


def build_cnn():
    """Same Sequential CNN architecture as ai_drp.py."""
    from tensorflow.keras.layers import (Conv2D, Dense, Flatten, MaxPooling2D)
    from tensorflow.keras.models import Sequential
    return Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(*IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])


def build_mobilenet():
    """Same MobileNetV2 transfer-learning model as ai_drp.py."""
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    preds = Dense(1, activation="sigmoid")(x)
    return Model(inputs=base.input, outputs=preds)


def train_model(model, train, val, epochs, name):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    model.fit(train, validation_data=val, epochs=epochs)
    loss, acc = model.evaluate(val, verbose=0)
    print(f"{name}: val_loss={loss:.4f} val_acc={acc:.4f}")
    return model


def train_resistance(csv_path):
    """RandomForest prototype on the TB trends CSV (same as ai_drp.py)."""
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    if not os.path.exists(csv_path):
        print(f"[skip] resistance model: {csv_path} not found")
        return None
    df = pd.read_csv(csv_path)
    feats = ["TB_Cases", "TB_Deaths", "TB_Treatment_Success_Rate"]
    missing = [c for c in feats if c not in df.columns]
    if missing or "TB_Incidence_Rate" not in df.columns:
        print(f"[skip] resistance model: missing columns {missing}")
        return None
    X = df[feats]
    y = df["TB_Incidence_Rate"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    res = RandomForestRegressor(random_state=42)
    res.fit(X_train, y_train)
    print(f"resistance model: train R^2={res.score(X_train, y_train):.3f} "
          f"test R^2={res.score(X_test, y_test):.3f}")
    return res


def main():
    p = argparse.ArgumentParser(description="Train TB models on real data (local).")
    p.add_argument("--data", default=DEFAULT_DATA, help="dataset root dir")
    p.add_argument("--trends", default=DEFAULT_TRENDS, help="TB trends CSV path")
    p.add_argument("--epochs", type=int, default=10, help="training epochs")
    p.add_argument("--no-mobilenet", action="store_true", help="skip MobileNetV2 model")
    args = p.parse_args()

    data_dir = find_data_dir(args.data)
    if not data_dir:
        print(f"ERROR: no TB dataset found at '{args.data}'.")
        print("Download it first (see samples/README.md):")
        print("  kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
        print("  unzip -q tuberculosis-tb-chest-xray-dataset.zip")
        print(f"Expected: {DEFAULT_DATA}/Normal/ and {DEFAULT_DATA}/Tuberculosis/")
        return 1

    print(f"dataset: {data_dir}")
    train, val = make_generators(data_dir)

    print("\n=== Training from-scratch CNN ->", CNN_MODEL_PATH, "===")
    cnn = train_model(build_cnn(), train, val, args.epochs, "CNN")
    cnn.save(CNN_MODEL_PATH)
    print(f"saved -> {CNN_MODEL_PATH}")

    if not args.no_mobilenet:
        print("\n=== Training MobileNetV2 transfer model ->", TL_MODEL_PATH, "===")
        tl = train_model(build_mobilenet(), train, val, args.epochs, "MobileNetV2")
        tl.save(TL_MODEL_PATH)
        print(f"saved -> {TL_MODEL_PATH}")

    print("\n=== Resistance prototype (RandomForest) ===")
    train_resistance(args.trends)
    print("\nDone. Run `python app.py` to use the trained model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
