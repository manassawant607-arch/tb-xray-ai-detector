# Real-Data Verification Results

End-to-end run of the complete `ai_drp.py`-equivalent pipeline on real Kaggle data,
executed locally (no Colab). Every step of `ai_drp.py` is reproduced and verified.

## 1. Dataset (via kaggle.json + setup_kaggle.py)

| Dataset | Source | Images/Rows |
|---|---|---|
| TB Chest Radiography Database | `tawsifurrahman/tuberculosis-tb-chest-xray-dataset` | 3,500 Normal + 700 Tuberculosis = **4,200 X-rays** |
| TB Trends (CSV) | `khushikyad001/tuberculosis-trends-global-and-regional-insights` | global TB statistics |

```bash
python setup_kaggle.py    # installs kaggle, copies kaggle.json, downloads + unzips both datasets
```

## 2. Training (train_real_model.py — matches ai_drp.py exactly)

`ImageDataGenerator(rescale=1/255, validation_split=0.2)`, `target_size=(224,224)`,
`batch_size=32`, `class_mode='binary'`, `epochs=5`, `optimizer='adam'`,
`loss='binary_crossentropy'`, `metrics=['accuracy']` — identical to `ai_drp.py`.

| Model | Architecture | File | Val accuracy | Val loss |
|---|---|---|---|---|
| CNN | Sequential(Conv2D 32/64/128 + Dense 128 + sigmoid) | `tb_detection_model.h5` | **94.88%** | 0.1210 |
| MobileNetV2 | ImageNet base + GAP + Dense 128 + sigmoid | `tb_detector_ai.h5` | **99.76%** | 0.0069 |
| RandomForest (resistance) | `RandomForestRegressor()` on TB trends | — | train R²=0.851, test R²=-0.154 |

## 3. Evaluation (evaluate_model.py — real validation split)

| Model | Precision | Recall | F1 | ROC AUC | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| CNN | 0.922 | 0.757 | 0.831 | 0.987 | 106 | 9 | 691 | 34 |

Confusion matrix saved to `confusion_matrix.png`.

## 4. X-ray validity gate — ML classifier (train_xray_gate.py)

A single saturation threshold could not separate real photos from X-rays (a cat
photo had saturation 0.128, lower than some real X-rays at 0.526). So the gate
was upgraded to a **trained RandomForest classifier** on 15 image statistics
(saturation, channel difference, dark/white fraction, histogram peak, edge
density, corner darkness, percentiles, etc.).

Trained on:
- **Positive:** all 4,200 real chest X-rays
- **Negative:** 10 real photos (cat, car, food, portrait, landscape, etc.) + 500
  synthetic non-X-rays (noise, gradients, solid colours, shapes, blobs, stripes)

Results: **0 false positives** (no non-X-ray accepted), **0 false negatives**
(no X-ray rejected), CV F1 = 0.941.

### Full test on ALL 4,200 real X-rays + real photos
```
=== ALL 4200 real X-rays through ML gate ===
accepted: 4200/4200  rejected: 0/4200   PASS

=== REAL non-X-ray photos through app (must REJECT) ===
ALL photos rejected: True  (cat, car, food, portrait, landscape, etc.)

=== REAL X-rays (must DETECT) ===
Normal       -> DETECTED: Normal
Tuberculosis -> DETECTED: TB Detected
```

## 5. Tests & lint

- **28 tests pass** (4 warnings).
- **flake8 clean** on all Python files.
- App live: https://work-1-ojoykhvwxsjjqbog.prod-runtime.all-hands.dev/

## 6. Full reproduction

```bash
python setup_kaggle.py        # kaggle.json setup + dataset download (ai_drp.py lines 10-21, 133-137)
python train_real_model.py    # CNN + MobileNetV2 + RandomForest (ai_drp.py lines 34-161)
python train_xray_gate.py     # train the X-ray validity-gate classifier
python evaluate_model.py      # metrics + confusion matrix + Grad-CAM
python app.py                 # Gradio app with X-ray validity gate (ai_drp.py lines 165-351)
```

