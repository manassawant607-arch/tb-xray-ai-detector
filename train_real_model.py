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


def make_generators(data_dir, seed=42, augment=False):
    """Train/val generators. Augmentation (flip/rotation/zoom/shift) on train only."""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_cfg = dict(rescale=1.0 / 255, validation_split=0.2)
    if augment:
        train_cfg.update(dict(
            horizontal_flip=True, rotation_range=10, zoom_range=0.1,
            width_shift_range=0.1, height_shift_range=0.1))
    train = ImageDataGenerator(**train_cfg).flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", subset="training", seed=seed)
    val = ImageDataGenerator(
        rescale=1.0 / 255, validation_split=0.2).flow_from_directory(
        data_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", subset="validation", seed=seed, shuffle=False)
    print("class indices:", train.class_indices)
    return train, val


def class_weights_from_generator(gen):
    """Balanced class weights for an imbalanced TB dataset (Normal >> TB)."""
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    labels = gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return dict(zip(classes.tolist(), weights.tolist()))


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


METRICS = ["accuracy"]
EXTRA_METRICS = ["precision", "recall", "auc"]


def train_model(model, train, val, epochs, name,
                class_weight=None, early_stop=False, extra_metrics=False):
    """Compile + fit. Defaults mirror ai_drp.py exactly (accuracy, no callbacks).

    Opt-in enhancements: --early-stop (EarlyStopping+ModelCheckpoint),
    --balance (class weights), and extra_metrics (precision/recall/auc).
    """
    from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint)
    metrics = METRICS + (EXTRA_METRICS if extra_metrics else [])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=metrics)
    model.summary()
    cbs = []
    if early_stop:
        cbs = [
            EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
            ModelCheckpoint(f"{name}_best.keras", monitor="val_loss",
                            save_best_only=True, verbose=0),
        ]
    model.fit(train, validation_data=val, epochs=epochs,
              callbacks=cbs or None, class_weight=class_weight)
    results = model.evaluate(val, verbose=0, return_dict=True)
    print(f"{name}: " + "  ".join(f"{k}={v:.4f}" for k, v in results.items()))
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
    print("trends columns:", df.columns.tolist())
    feats = ["TB_Cases", "TB_Deaths", "TB_Treatment_Success_Rate"]
    missing = [c for c in feats if c not in df.columns]
    if missing or "TB_Incidence_Rate" not in df.columns:
        print(f"[skip] resistance model: missing columns {missing}")
        return None
    X = df[feats]
    y = df["TB_Incidence_Rate"]
    # ai_drp.py: train_test_split(X, y, test_size=0.2) + RandomForestRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    res = RandomForestRegressor()
    res.fit(X_train, y_train)
    print(f"resistance model: train R^2={res.score(X_train, y_train):.3f} "
          f"test R^2={res.score(X_test, y_test):.3f}")
    return res


def main():
    p = argparse.ArgumentParser(description="Train TB models on real data (local).")
    p.add_argument("--data", default=DEFAULT_DATA, help="dataset root dir")
    p.add_argument("--trends", default=DEFAULT_TRENDS, help="TB trends CSV path")
    p.add_argument("--epochs", type=int, default=5,
                   help="training epochs (ai_drp.py uses 5)")
    p.add_argument("--no-mobilenet", action="store_true", help="skip MobileNetV2 model")
    p.add_argument("--augment", action="store_true",
                   help="training-time augmentation (off by default, matches ai_drp.py)")
    p.add_argument("--balance", action="store_true",
                   help="balanced class weights (off by default, matches ai_drp.py)")
    p.add_argument("--early-stop", action="store_true",
                   help="EarlyStopping + ModelCheckpoint (off by default)")
    p.add_argument("--metrics", action="store_true",
                   help="also track precision/recall/auc (off by default)")
    args = p.parse_args()

    data_dir = find_data_dir(args.data)
    if not data_dir:
        print(f"ERROR: no TB dataset found at '{args.data}'.")
        print("Download it first (see samples/README.md):")
        print("  kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
        print("  unzip -q tuberculosis-tb-chest-xray-dataset.zip")
        print(f"Expected: {DEFAULT_DATA}/Normal/ and {DEFAULT_DATA}/Tuberculosis/")
        return 1

    print(f"dataset: {data_dir}  epochs={args.epochs}  "
          f"augment={args.augment}  balance={args.balance}")
    train, val = make_generators(data_dir, augment=args.augment)
    cw = class_weights_from_generator(train) if args.balance else None
    if cw:
        print("class weights:", cw)

    print("\n=== Training from-scratch CNN ->", CNN_MODEL_PATH, "===")
    cnn = train_model(build_cnn(), train, val, args.epochs, "CNN",
                      class_weight=cw, early_stop=args.early_stop,
                      extra_metrics=args.metrics)
    cnn.save(CNN_MODEL_PATH)
    print(f"saved -> {CNN_MODEL_PATH}")

    if not args.no_mobilenet:
        print("\n=== Training MobileNetV2 transfer model ->", TL_MODEL_PATH, "===")
        tl = train_model(build_mobilenet(), train, val, args.epochs,
                         "MobileNetV2",
                         class_weight=cw, early_stop=args.early_stop,
                         extra_metrics=args.metrics)
        tl.save(TL_MODEL_PATH)
        print(f"saved -> {TL_MODEL_PATH}")

    print("\n=== Resistance prototype (RandomForest) ===")
    train_resistance(args.trends)
    print("\nDone. Run `python app.py` to use the trained model,")
    print("or `python evaluate_model.py` for metrics + confusion matrix + Grad-CAM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
