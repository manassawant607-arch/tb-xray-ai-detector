# TB X-Ray AI Detector

AI-powered Tuberculosis (TB) chest X-ray detector and prototype drug-resistance/treatment suggestion UI built with TensorFlow and Gradio.

This repository contains a lightweight Gradio app (app.py) and a Colab-derived training / prototype notebook script (copy_of_ai_drp (1).py). The trained Keras model file (tb_detection_model.h5) is expected to be placed in the repository root if you want to run inference locally.

---

## Repository structure

- app.py
  - Minimal Gradio app that exposes a detect_tb(image) function. Intended as a quick demo UI.
- copy_of_ai_drp (1).py
  - Colab-exported script containing dataset download (Kaggle), data preprocessing, two model definitions (from-scratch and MobileNetV2 transfer learning), training, model saving, and several Gradio demo variations.
- requirements.txt
  - Python packages required to run the demos: gradio, tensorflow, numpy, pillow


## Quickstart (inference/demo)

1. Clone the repository:

   git clone https://github.com/manassawant607-arch/tb-xray-ai-detector.git
   cd tb-xray-ai-detector

2. Create a virtual environment (recommended) and install dependencies:

   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .\.venv\Scripts\activate   # Windows (PowerShell)

   pip install -r requirements.txt

3. Prepare the trained model file

   - The demo scripts expect a trained Keras model named `tb_detection_model.h5` in the repository root.
   - If you don't have one, see the Training section below or download one you trained earlier and place it at the root as `tb_detection_model.h5`.

4. Run the simple demo UI:

   python app.py

   - This will launch a Gradio web interface. Upload a chest X-ray image to see a placeholder response or, if a model is present, the model's prediction (depending on which script you run).


## Training (Colab / Kaggle)

The file `copy_of_ai_drp (1).py` contains the steps used in a Colab environment to train a TB detection model. Key points:

- Dataset: TB Chest Radiography Database (examples in the script use `TB_Chest_Radiography_Database`). The script also uses Kaggle to download datasets — you must provide a `kaggle.json` with your Kaggle API credentials when running the Colab notebook or local Kaggle commands.
- Two approaches shown:
  - A simple CNN (Sequential) trained from scratch.
  - Transfer learning with MobileNetV2 (recommended for better performance and faster convergence).
- Training parameters in the script are minimal (e.g., 5 epochs). For production or meaningful results, increase epochs, use proper data augmentation, class balancing, and utilize GPUs.
- Saved model filenames used in the script: `tb_detection_model.h5`, `tb_detector_ai.h5`.

If you plan to reproduce training locally, install required packages, ensure you have the dataset directory structured with class subfolders (e.g., `TB_Chest_Radiography_Database/train/Tuberculosis` and `/Normal`), and train using the notebook or script.


## Running the full prototype

Several Gradio interfaces are present in the Colab script demonstrating different outputs (single text, multiple textboxes with mutation/resistance/treatment). To run them locally:

- Rename or run the `copy_of_ai_drp (1).py` file from a safe path (avoid spaces in filename; e.g., rename to `train_and_demo.py`).
- Ensure `tb_detection_model.h5` exists in the repository root (or retrain to create it).
- Run the script: `python train_and_demo.py` and follow the Gradio link printed in the console.


## Notes & Caveats

- Medical disclaimer: This project is a research/prototype demo and NOT a medical device. Do NOT use this system for diagnosis or treatment decisions. Always consult a qualified healthcare professional.
- Model performance: The script and example models in this repository are simple starting points. Real-world TB detection requires rigorous dataset curation, validation, explainability, and clinical trials.
- Privacy: Chest X-ray images may be sensitive medical data. Handle patient data according to applicable laws and institutional policies.


## Suggestions / Next steps

- Add a proper notebook (Jupyter / Colab) with cleaned training steps and parameter tuning.
- Add automated tests and a CI workflow for linting and dependency checks.
- Add a small sample of anonymized X-rays for a quick local demo (if licensing allows) or document how to download the Kaggle dataset.
- Replace placeholder messages in `app.py` with the inference code used in the Colab script (preprocessing load model predict mapping to labels).


## Contact

If you want, I can:
- Rename and clean up the Colab script (remove duplicated blocks, fix filename with spaces).
- Add the inference loading logic to `app.py` so the simple Gradio app uses `tb_detection_model.h5` if present.
- Open a PR with these changes directly in this repository.

Reply which of the above you want me to do and I will continue.


## License

Include a license as appropriate for your project (for example, MIT). Add a LICENSE file to the repo if you want me to create one.


---

Disclaimer: For development and research only — not for clinical use.
