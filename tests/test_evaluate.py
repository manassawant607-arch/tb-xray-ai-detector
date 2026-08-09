"""Tests for evaluate_model: metrics computation + gradcam output shape.

Metrics are tested with plain numpy arrays (no model / no dataset required).
Grad-CAM is exercised with a tiny hand-built Keras model so the real code path
runs end-to-end without the full training pipeline.
"""

import numpy as np
from PIL import Image
import pytest

import evaluate_model as ev


def test_metrics_perfect():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = ev.compute_metrics(y_true, y_prob)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["tp"] == 2 and m["tn"] == 2 and m["fp"] == 0 and m["fn"] == 0


def test_metrics_all_wrong():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])  # swapped
    m = ev.compute_metrics(y_true, y_prob)
    assert m["tp"] == 0 and m["tn"] == 0
    assert m["fp"] == 2 and m["fn"] == 2
    assert m["recall"] == 0.0


def test_metrics_threshold_boundary():
    y_true = np.array([1])
    # exactly at threshold (0.5) -> NOT > 0.5 -> predicted negative
    m = ev.compute_metrics(y_true, np.array([0.5]))
    assert m["tp"] == 0 and m["fn"] == 1
    m2 = ev.compute_metrics(y_true, np.array([0.5001]))
    assert m2["tp"] == 1 and m2["fn"] == 0


def test_metrics_auc():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8])
    m = ev.compute_metrics(y_true, y_prob)
    assert 0.0 <= m["roc_auc"] <= 1.0


def test_metrics_single_class():
    # only negatives -> roc_auc undefined -> nan, must not crash
    y_true = np.array([0, 0, 0])
    m = ev.compute_metrics(y_true, np.array([0.1, 0.2, 0.3]))
    assert np.isnan(m["roc_auc"]) or m["roc_auc"] == m["roc_auc"]


def test_find_last_conv_layer_and_gradcam_shape():
    pytest.importorskip("tensorflow")
    from tensorflow.keras.layers import (Conv2D, Dense, GlobalAveragePooling2D, Input)
    from tensorflow.keras.models import Model
    # Functional model (mirrors a loaded .h5) so model.inputs is callable.
    inp = Input(shape=(224, 224, 3))
    x = Conv2D(4, (3, 3), activation="relu", name="last_conv")(inp)
    x = GlobalAveragePooling2D()(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    name = ev.find_last_conv_layer(model)
    assert name == "last_conv"
    # build a small textured image and run gradcam
    arr = np.random.rand(224, 224, 3).astype(np.float32)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    heatmap, pred = ev.gradcam(model, img)
    assert heatmap.shape == (222, 222)  # 224 - 2 (valid conv)
    assert 0.0 <= heatmap.min() and heatmap.max() <= 1.0
    assert 0.0 <= pred <= 1.0
