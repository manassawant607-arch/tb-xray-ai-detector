"""Tests for xray_validator: right photos (valid X-rays) accepted, wrong photos rejected.

A "wrong photo" (colourful photo, blank, tiny, extreme aspect) must NOT pass the
gate, so it can never be falsely detected as TB. The synthetic sample X-rays in
samples/ must pass.
"""

import numpy as np
from PIL import Image

import xray_validator as xv


def make_gray_xray(size=(512, 512)):
    """A textured, near-grayscale image that mimics a chest X-ray."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w // 2, h // 2
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (w * 0.18) ** 2)))
    img = np.full((h, w), 120, dtype=np.float32)
    img += blob * 40
    img += np.random.default_rng(0).normal(0, 10, (h, w))
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img, mode="L").convert("RGB")


def make_colour_photo(size=(512, 512)):
    """A colourful image (high saturation) — not an X-ray."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = (xx / w * 255).astype(np.uint8)          # red gradient
    arr[..., 1] = (yy / h * 255).astype(np.uint8)          # green gradient
    arr[..., 2] = ((1 - xx / w) * 255).astype(np.uint8)    # blue gradient
    return Image.fromarray(arr, mode="RGB")


def make_blank_white(size=(512, 512)):
    return Image.new("RGB", size, (255, 255, 255))


def make_blank_black(size=(512, 512)):
    return Image.new("RGB", size, (0, 0, 0))


def make_tiny(size=(30, 30)):
    return make_gray_xray(size)


def make_strip(size=(1200, 120)):
    return make_gray_xray(size)


def test_accepts_synthetic_normal_sample():
    img = Image.open("samples/sample_normal.png")
    ok, reason = xv.is_chest_xray(img)
    assert ok, f"expected normal sample to be a valid X-ray: {reason}"


def test_accepts_synthetic_tb_sample():
    img = Image.open("samples/sample_tb_like.png")
    ok, reason = xv.is_chest_xray(img)
    assert ok, f"expected tb sample to be a valid X-ray: {reason}"


def test_accepts_textured_grayscale():
    ok, _ = xv.is_chest_xray(make_gray_xray())
    assert ok


def test_rejects_colour_photo():
    ok, reason = xv.is_chest_xray(make_colour_photo())
    assert not ok
    assert "colourful" in reason


def test_rejects_blank_white():
    ok, reason = xv.is_chest_xray(make_blank_white())
    assert not ok


def test_rejects_blank_black():
    ok, reason = xv.is_chest_xray(make_blank_black())
    assert not ok


def test_rejects_tiny_image():
    ok, reason = xv.is_chest_xray(make_tiny())
    assert not ok
    assert "too small" in reason


def test_rejects_extreme_aspect():
    ok, reason = xv.is_chest_xray(make_strip())
    assert not ok
    assert "aspect" in reason


def test_rejects_none():
    ok, reason = xv.is_chest_xray(None)
    assert not ok


def test_saturation_grayscale_near_zero():
    arr = np.asarray(make_gray_xray()).astype(np.float32) / 255.0
    assert xv.saturation(arr) < 0.05


def test_saturation_colour_high():
    arr = np.asarray(make_colour_photo()).astype(np.float32) / 255.0
    assert xv.saturation(arr) > 0.4


def test_returns_reason_tuple():
    ok, reason = xv.is_chest_xray(make_gray_xray())
    assert isinstance(ok, bool)
    assert isinstance(reason, str) and reason
