# TB X-Ray AI Detector

AI-powered Tuberculosis (TB) chest X-ray detection prototype with a drug-resistance / treatment-suggestion UI, built with **TensorFlow** and **Gradio**.

> 🚀 **Live demo:** <https://work-1-ojoykhvwxsjjqbog.prod-runtime.all-hands.dev/>
>
> Upload a chest X-ray and get **TB Detection**, **Mutation Analysis**, **Drug Resistance Prediction**, **Treatment Recommendation**, and a **Confidence** score. Try the built-in example images under the input. Currently running in **inference mode** with a trained CNN model.

> ⚠️ **Medical disclaimer:** This project is a research / educational prototype and is **NOT** a medical device. Do **not** use it for diagnosis or treatment decisions. Always consult a qualified healthcare professional.

---

## Overview

The project detects Tuberculosis from chest X-ray images using a trained Keras model and presents a prototype pipeline that adds mutation analysis, drug-resistance prediction, and treatment recommendations.

Two entry points are provided:

| File | Purpose |
| --- | --- |
| `app.py` | Gradio web app. Takes an X-ray image and returns TB detection, mutation, drug-resistance, treatment, and confidence. Runs real inference if `tb_detection_model.h5` is present, otherwise demo mode. |
| `tb_inference.py` | Shared, testable inference logic (preprocessing, prediction interpretation, confidence, model loading, demo fallback). |
| `train_demo_model.py` | Trains a tiny CNN on **synthetic** image data to produce `tb_detection_model.h5`, so the inference path runs end-to-end without the real dataset. |
| `train_real_model.py` | Trains on **real** TB data locally — mirrors the `ai_drp.py` pipeline (same CNN + MobileNetV2 + RandomForest resistance prototype) but runs outside Colab. |
| `setup_kaggle.py` | Sets up `kaggle.json` credentials and downloads + unzips both real datasets used by `ai_drp.py` (TB X-rays + trends CSV). |
| `evaluate_model.py` | Computes precision/recall/F1/AUC, saves a confusion matrix, and generates a Grad-CAM heatmap on the real validation set. |
| `REAL_DATA_RESULTS.md` | Full verification log of the real-data run (training, evaluation, X-ray validity gate on all 4,200 images). |
| `xray_validator.py` | Two-layer validity gate: heuristic pre-filter + trained RandomForest classifier that rejects non-X-ray images (colour photos, screenshots, blanks, tiny) so "wrong photos" are never falsely detected as TB. |
| `train_xray_gate.py` | Trains the X-ray validity-gate RandomForest on 4,200 real X-rays + non-X-ray negatives (saves `xray_gate_model.pkl`). |
| `ai_drp.py` | Colab-exported script containing the original full workflow (dataset download, preprocessing, two model definitions, training, demos). |
| `ai_drp.ipynb` | Cleaned, parameterized Jupyter/Colab notebook of the same workflow with tunable config, augmentation, and Dropout. |
| `tests/test_inference.py` | Unit tests for preprocessing, prediction interpretation, confidence, and the demo fallback (run without a trained model). |
| `samples/` | Synthetic placeholder images + docs on downloading the real Kaggle dataset. |
| `requirements.txt` | Python dependencies: `gradio`, `tensorflow`, `numpy`, `pillow`. |

A trained Keras model named `tb_detection_model.h5` is expected in the repository root to run inference.

## Repository structure

```
tb-xray-ai-detector/
├── app.py              # Gradio web app (inference + demo fallback)
├── tb_inference.py     # Shared, testable inference logic
├── train_demo_model.py # Train a synthetic-data model for the inference path
├── ai_drp.py           # Original Colab training script
├── ai_drp.ipynb        # Cleaned, parameterized notebook
├── requirements.txt    # Python dependencies
├── tests/              # pytest unit tests
├── samples/            # Synthetic placeholder images + dataset docs
├── .github/workflows/  # CI (lint + test + dependency check)
├── .flake8             # Lint config
└── README.md
```

---

## Quickstart

### Try the live demo (no setup)

A hosted instance is already running — no clone or install required:

> 🚀 **<https://work-1-ojoykhvwxsjjqbog.prod-runtime.all-hands.dev/>**

1. Open the link in your browser.
2. Click **"Click to Upload"** (or drag) a chest X-ray image, or use one of the
   built-in **Examples** below the input.
3. Click **Submit**.
4. Read the five outputs: **TB Detection**, **Mutation Analysis**, **Drug
   Resistance Prediction**, **Treatment Recommendation**, and **Confidence**.

The banner above the input shows whether the app is in
`✅ Inference mode` (a trained model is loaded) or `⚠️ Demo mode` (placeholder
output). You can also call it programmatically via the **Use via API** button.

### 1. Clone the repository

```bash
git clone https://github.com/manassawant607-arch/tb-xray-ai-detector.git
cd tb-xray-ai-detector
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .\.venv\Scripts\activate  # Windows (PowerShell)

pip install -r requirements.txt
```

### 3. Prepare the trained model

The app expects a trained Keras model named `tb_detection_model.h5` in the repository root.

- If you already have a trained model, place it at the repo root as `tb_detection_model.h5`.
- To quickly enable the real-inference path **without the real dataset**, train a tiny CNN on synthetic data:

  ```bash
  python train_demo_model.py    # writes tb_detection_model.h5 (~1 min on CPU)
  ```

  > This model is trained on class-correlated synthetic gradients — it makes the
  > inference code path run end-to-end but is **not** medically meaningful.
- To train on **real data locally** (mirrors the `ai_drp.py` Colab pipeline),
  first download the Kaggle TB Chest Radiography Database, then run:

  ```bash
  kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset
  unzip -q tuberculosis-tb-chest-xray-dataset.zip   # -> TB_Chest_Radiography_Database/
  python train_real_model.py --epochs 10             # -> tb_detection_model.h5 (+ tb_detector_ai.h5)
  ```

  `train_real_model.py` uses the same Sequential CNN, the same MobileNetV2
  transfer-learning model, and the same RandomForest resistance prototype as
  `ai_drp.py`, but runs locally (no `google.colab` / `!` shell commands).
- For a real model, train on the actual TB dataset following the
  [Training](#training-colab--kaggle) section below.

### 4. Run the demo

```bash
python app.py
```

This launches a Gradio web interface (default port `12000`) that returns TB
detection, mutation, drug-resistance, treatment, and a **confidence** score.
If `tb_detection_model.h5` is present it runs **real inference**; otherwise it
starts in **demo mode** with clearly-labelled placeholder output. The UI ships
with built-in examples from `samples/`. Try it with the synthetic images there.

CLI options:

```bash
python app.py --port 12000   # fixed port
python app.py --share         # public Gradio share link
```

### 5. Run the tests

```bash
pip install pytest flake8
pytest -q                     # unit tests (no model required)
flake8 .                      # lint
```

---

## How it works

The `ai_drp.py` script implements the following pipeline:

1. **Dataset download** — Uses Kaggle to fetch the *TB Chest Radiography Database* (`tawsifurrahman/tuberculosis-tb-chest-xray-dataset`). A `kaggle.json` with your Kaggle API credentials is required.
2. **Preprocessing** — `ImageDataGenerator` rescales images to `[0,1]` and creates an 80/20 train/validation split at `224x224` resolution.
3. **Training (two models)**
   - A **from-scratch CNN** (`Sequential` with Conv2D/MaxPooling/Dense layers) saved as `tb_detection_model.h5`.
   - **Transfer learning with MobileNetV2** (ImageNet weights, frozen base + custom head) saved as `tb_detector_ai.h5`.
4. **Resistance trends (prototype)** — A `RandomForestRegressor` trained on a TB trends dataset (`khushikyad001/tuberculosis-trends-global-and-regional-insights`) to estimate TB incidence risk.
5. **Gradio demos** — Several interfaces ranging from a simple text result to a four-output panel: **TB Detection**, **Mutation Analysis**, **Drug Resistance Prediction**, and **Treatment Recommendation**.

### Inference logic

```python
def predict_tb(image):
    img = preprocess(image)                       # resize -> 224x224, /255, expand dims
    pred = model.predict(img)[0][0]               # sigmoid output in [0,1]

    if pred > 0.5:
        tb         = "TB Detected"
        mutation   = "rpoB mutation detected"
        resistance = "Rifampicin Resistant (Possible MDR-TB)"
        treatment  = "Bedaquiline + Linezolid + Levofloxacin"
    else:
        tb         = "Normal"
        mutation   = "No mutation detected"
        resistance = "Drug Sensitive"
        treatment  = "Standard TB therapy"

    return tb, mutation, resistance, treatment
```

> The mutation / resistance / treatment outputs are **prototype placeholders** for demonstration and are not derived from clinical models.

### Chest X-ray validity gate (only X-rays are detected)

A **wrong photo** (a colourful photograph, a screenshot, a blank or tiny image) is
rejected *before* detection, so it can never be falsely flagged as TB. Only a
**right photo** (a valid chest X-ray) flows into the detection model.

`xray_validator.py` uses a two-layer gate:

1. **Heuristic pre-filter** (no model needed) — rejects obvious junk:
   | Check | Rejects |
   |---|---|
   | Minimum dimension (`MIN_DIM=100`) | postage-stamp / tiny images |
   | Aspect ratio (`MAX_ASPECT=2.5`) | very wide or tall strips |
   | Brightness range (`0.04`–`0.985`) | pure-black / pure-white / blank images |
   | Texture / std (`>0.015`) | flat uniform images (solid color blocks) |

2. **Trained RandomForest classifier** (`xray_gate_model.pkl`) — 15 image
   statistics (saturation, channel difference, dark/white fraction, histogram
   peak, edge density, corner darkness, percentiles, etc.). This catches
   real-world photos that pass the heuristics — a single saturation threshold
   cannot separate them (a cat photo had saturation 0.128, lower than some real
   X-rays at 0.526). Trained by `train_xray_gate.py` on all 4,200 real X-rays
   + 510 non-X-ray negatives. Result: **0 false positives, 0 false negatives**.

Verified end-to-end: **4200/4200 real X-rays accepted & predicted**; all real
photos (cat, car, food, portrait, landscape) and all synthetic wrong photos
rejected.

### Local training on real data

`setup_kaggle.py` + `train_real_model.py` reproduce the **entire `ai_drp.py`
pipeline locally** (no Colab needed). Every step of `ai_drp.py` is covered:

| `ai_drp.py` step | Lines | Local equivalent |
|---|---|---|
| `!pip install kaggle`, copy `kaggle.json`, chmod 600 | 10–17 | `python setup_kaggle.py` |
| Download + unzip TB chest X-ray dataset | 19–21 | `python setup_kaggle.py` |
| `ImageDataGenerator` rescale + 80/20 split, 224×224 | 34–57 | `train_real_model.make_generators()` |
| from-scratch CNN → `tb_detection_model.h5` | 59–93 | `train_real_model.build_cnn()` + `train_model()` |
| MobileNetV2 transfer → `tb_detector_ai.h5` | 95–131 | `train_real_model.build_mobilenet()` |
| Download trends CSV + RandomForest resistance | 133–161 | `setup_kaggle.py` + `train_resistance()` |
| Gradio TB/mutation/resistance/treatment UI | 163–426 | `app.py` (with X-ray validity gate) |

`train_real_model.py` defaults match `ai_drp.py` exactly (epochs=5,
`metrics=['accuracy']`, no augmentation/class-weights/callbacks);
improvements are opt-in flags.

```bash
# 1. Place kaggle.json (from https://www.kaggle.com/settings -> API -> Create New Token)
#    in the repo root, then:
python setup_kaggle.py        # installs kaggle, sets up credentials, downloads + unzips both datasets

# 2. Train (matches ai_drp.py: epochs=5, both CNN + MobileNetV2)
python train_real_model.py
# opt-in enhancements:
python train_real_model.py --augment --balance --early-stop --metrics --epochs 15
```

**Real-data results** (3,500 Normal + 700 Tuberculosis X-rays, 80/20 split):
see [`REAL_DATA_RESULTS.md`](REAL_DATA_RESULTS.md) for the full verification.

| Model | File | Val accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|
| CNN | `tb_detection_model.h5` | 94.88% | 0.922 | 0.757 | 0.831 | 0.987 |
| MobileNetV2 | `tb_detector_ai.h5` | 99.76% | 0.986 | 0.993 | 0.989 | 1.000 |

`evaluate_model.py` computes precision/recall/F1/AUC, saves a confusion-matrix
PNG, and can generate a Grad-CAM heatmap:

```bash
python evaluate_model.py                              # CNN metrics + confusion matrix
python evaluate_model.py --model tb_detector_ai.h5    # MobileNetV2
python evaluate_model.py --gradcam some_xray.png      # explainability heatmap
```

---

## Training (Colab / Kaggle)

`ai_drp.py` was exported from a Colab notebook and contains the steps used to train the TB detection model. To reproduce:

1. Open the script in [Google Colab](https://colab.research.google.com/) (recommended, since it uses `google.colab` helpers and `!` shell commands).
2. Upload your `kaggle.json` (Kaggle API token) when prompted.
3. Run the cells in order to:
   - Download and unzip the *TB Chest Radiography Database*.
   - Preprocess the data and train the CNN and/or MobileNetV2 models.
   - Save `tb_detection_model.h5` / `tb_detector_ai.h5`.
4. Download the saved model and place it in the repository root for local inference.

### Notes on training parameters

- The default training uses **5 epochs** for quick prototyping. For meaningful results, increase epochs, add data augmentation, balance classes, and use a GPU.
- Ensure the dataset directory is structured with class subfolders (e.g., `TB_Chest_Radiography_Database/Normal/` and `TB_Chest_Radiography_Database/Tuberculosis/`).

---

## Running the full prototype

`ai_drp.ipynb` is the cleaned, parameterized version of the workflow (config
cell with `EPOCHS_*`, `IMG_SIZE`, `BATCH_SIZE`; added augmentation and Dropout).
Open it in [Colab](https://colab.research.google.com/) and run the cells in order.

`ai_drp.py` is the original raw Colab export and contains several duplicated
Gradio interfaces. To run a specific demo from it locally:

1. Ensure `tb_detection_model.h5` exists in the repository root (or retrain to create it).
2. Open `ai_drp.py`, keep the interface block you want to run, and comment out the others (the file contains multiple `app.launch()` / `demo.launch()` calls).
3. Run the script:

   ```bash
   python ai_drp.py
   ```

4. Open the Gradio link printed in the console.

> Both files contain Colab-specific shell commands (`!pip`, `!kaggle`, `from google.colab import files`) that only run inside Colab. Strip or adapt those lines if running purely locally — for local inference prefer `app.py`, which already handles the model-loading fallback.

---

## Notes & caveats

- **Not a medical device.** This is a prototype for research/education only.
- **Model performance.** The included model definitions are simple starting points. Real-world TB detection requires rigorous dataset curation, validation, explainability, and clinical trials.
- **Prototype outputs.** The mutation, resistance, and treatment outputs are illustrative placeholders, not clinically validated predictions.
- **Privacy.** Chest X-ray images may be sensitive medical data. Handle patient data according to applicable laws and institutional policies.

---

## Testing & CI

### Unit tests

`tests/test_inference.py` covers the testable parts of the pipeline **without
requiring a trained model** (so CI stays fast):

- preprocessing output shape/range and grayscale→RGB conversion
- prediction interpretation at positive, negative, and threshold boundaries
- `model_available()` for a missing path
- demo fallback returns 4 non-empty fields (including the `None` image case)

```bash
pytest -q
```

### Linting

```bash
flake8 .          # config in .flake8; ai_drp.py excluded (raw Colab export)
```

### Continuous integration

`.github/workflows/ci.yml` runs on every push/PR to `main`:

1. **lint-and-test** — installs deps, runs `flake8`, runs `pytest`.
2. **dependency-check** — verifies `requirements.txt` installs cleanly and core imports succeed.

---

## License

This project is provided for research and educational purposes. Add a `LICENSE` file (e.g., MIT) as appropriate for your use case.

---

*Disclaimer: For development and research only — not for clinical use.*
