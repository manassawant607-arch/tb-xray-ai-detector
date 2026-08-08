# TB X-Ray AI Detector

AI-powered Tuberculosis (TB) chest X-ray detection prototype with a drug-resistance / treatment-suggestion UI, built with **TensorFlow** and **Gradio**.

> ⚠️ **Medical disclaimer:** This project is a research / educational prototype and is **NOT** a medical device. Do **not** use it for diagnosis or treatment decisions. Always consult a qualified healthcare professional.

---

## Overview

The project detects Tuberculosis from chest X-ray images using a trained Keras model and presents a prototype pipeline that adds mutation analysis, drug-resistance prediction, and treatment recommendations.

Two entry points are provided:

| File | Purpose |
| --- | --- |
| `app.py` | Minimal Gradio demo app. Launches a web UI that takes an X-ray image and returns a placeholder response. |
| `ai_drp.py` | Colab-exported script containing the full workflow: dataset download, preprocessing, model training (from-scratch CNN + MobileNetV2 transfer learning), model saving, and several Gradio demo variations. |
| `requirements.txt` | Python dependencies: `gradio`, `tensorflow`, `numpy`, `pillow`. |

A trained Keras model named `tb_detection_model.h5` is expected in the repository root to run inference.

---

## Repository structure

```
tb-xray-ai-detector/
├── app.py              # Minimal Gradio demo UI
├── ai_drp.py           # Training + full prototype demos (Colab script)
├── requirements.txt    # Python dependencies
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

The demo scripts expect a trained Keras model named `tb_detection_model.h5` in the repository root.

- If you already have a trained model, place it at the repo root as `tb_detection_model.h5`.
- Otherwise, train one following the [Training](#training-colab--kaggle) section below.

### 4. Run the demo

```bash
python app.py
```

This launches a Gradio web interface. Upload a chest X-ray image to get a response. When a trained model is present, the `ai_drp.py` demos will return the model's prediction along with prototype resistance/treatment output.

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

`ai_drp.py` includes several Gradio interfaces demonstrating different output styles. To run a specific demo locally:

1. Ensure `tb_detection_model.h5` exists in the repository root (or retrain to create it).
2. Open `ai_drp.py`, keep the interface block you want to run, and remove or comment out the others (the file contains multiple `app.launch()` / `demo.launch()` calls as exported from Colab).
3. Run the script:

   ```bash
   python ai_drp.py
   ```

4. Open the Gradio link printed in the console.

> Because `ai_drp.py` is a raw Colab export, it contains Colab-specific shell commands (`!pip`, `!kaggle`, `from google.colab import files`) that only run inside Colab. Strip or adapt those lines if running purely locally.

---

## Notes & caveats

- **Not a medical device.** This is a prototype for research/education only.
- **Model performance.** The included model definitions are simple starting points. Real-world TB detection requires rigorous dataset curation, validation, explainability, and clinical trials.
- **Prototype outputs.** The mutation, resistance, and treatment outputs are illustrative placeholders, not clinically validated predictions.
- **Privacy.** Chest X-ray images may be sensitive medical data. Handle patient data according to applicable laws and institutional policies.

---

## Possible next steps

- Clean up `ai_drp.py` into a proper Jupyter notebook with parameter tuning.
- Add the inference loading logic to `app.py` so the minimal app uses `tb_detection_model.h5` when present.
- Add automated tests and a CI workflow for linting and dependency checks.
- Add a small set of anonymized sample X-rays for a quick local demo (if licensing allows).

---

## License

This project is provided for research and educational purposes. Add a `LICENSE` file (e.g., MIT) as appropriate for your use case.

---

*Disclaimer: For development and research only — not for clinical use.*
