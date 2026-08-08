# TB X-Ray AI Detector

AI-powered Tuberculosis (TB) chest X-ray detection prototype with a drug-resistance / treatment-suggestion UI, built with **TensorFlow** and **Gradio**.

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
