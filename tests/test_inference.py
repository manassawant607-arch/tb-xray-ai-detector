"""Tests for tb_inference: preprocessing shape, prediction interpretation, demo fallback.

These run without a trained Keras model (demo/interpretation paths only) so CI
stays fast and dependency-light. Real-model inference is covered when a model
file is present via TB_MODEL_PATH.
"""

from PIL import Image

import tb_inference as tbi


def make_image(color=(0, 0, 0), size=(300, 300)):
    return Image.new("RGB", size, color)


def test_preprocess_shape_and_range():
    img = make_image()
    arr = tbi.preprocess(img)
    assert arr.shape == (1, 224, 224, 3)
    assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_preprocess_converts_grayscale_to_rgb():
    gray = Image.new("L", (224, 224), 128)
    arr = tbi.preprocess(gray)
    assert arr.shape == (1, 224, 224, 3)


def test_interpret_positive():
    assert tbi.interpret_prediction(0.9) == tbi.TB_POSITIVE


def test_interpret_negative():
    assert tbi.interpret_prediction(0.1) == tbi.TB_NEGATIVE


def test_interpret_threshold_boundary():
    # threshold is strict: exactly 0.5 is NOT > 0.5, so it maps to negative
    assert tbi.interpret_prediction(0.5) == tbi.TB_NEGATIVE
    assert tbi.interpret_prediction(0.5001) == tbi.TB_POSITIVE


def test_model_available_false_for_missing(tmp_path):
    assert tbi.model_available(str(tmp_path / "nope.h5")) is False


def test_demo_predict_returns_five_fields():
    out = tbi.demo_predict(make_image())
    assert len(out) == 5
    assert all(isinstance(field, str) and field for field in out)
    assert out[0].startswith("[DEMO]")


def test_demo_predict_none_image():
    out = tbi.demo_predict(None)
    assert len(out) == 5
    assert "[DEMO]" in out[0]


def test_confidence_format():
    assert tbi.confidence(0.9) == "90.0%"
    assert tbi.confidence(0.1) == "90.0%"
    assert tbi.confidence(0.5) == "50.0%"


def test_pipeline_labels_consistent():
    assert len(tbi.TB_POSITIVE) == len(tbi.TB_NEGATIVE) == 4
