"""TB X-Ray AI Detector Gradio app.

Launches a web UI that takes a chest X-ray image and returns TB detection,
mutation analysis, drug-resistance prediction, and a treatment recommendation.

If a trained Keras model (`tb_detection_model.h5`) is present in the repo root
(or at $TB_MODEL_PATH), real inference is run. Otherwise the app starts in
demo mode with clearly-labelled placeholder output.

Usage:
    python app.py                       # auto host/port
    python app.py --port 12000          # fixed port
    python app.py --share               # public Gradio share link
"""

import argparse
import os

import gradio as gr

import tb_inference as tbi
import xray_validator as xv


def predict(image):
    """Gradio callback: validate -> real inference if a model exists, else demo.

    A "wrong photo" (not a chest X-ray) is rejected here and is never detected,
    so it cannot be falsely flagged as TB. A "right photo" (valid X-ray) flows
    into the unchanged detection logic in tb_inference.py.
    """
    if image is None:
        return ("No image provided", "", "", "", "")

    ok, reason = xv.is_chest_xray(image)
    if not ok:
        return (reason, "", "", "", "—")

    if tbi.model_available():
        model = tbi.load_model()
        return tbi.predict_tb(image, model)
    return tbi.demo_predict(image)


def build_interface():
    model_ok = tbi.model_available()
    status = ("✅ Inference mode — trained model loaded."
              if model_ok
              else "⚠️ Demo mode — no trained model found; output is placeholder.")
    examples = (
        [["samples/sample_normal.png"], ["samples/sample_tb_like.png"]]
        if os.path.exists("samples/sample_normal.png") else None
    )
    return gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload Chest X-ray"),
        outputs=[
            gr.Textbox(label="TB Detection"),
            gr.Textbox(label="Mutation Analysis"),
            gr.Textbox(label="Drug Resistance Prediction"),
            gr.Textbox(label="Treatment Recommendation"),
            gr.Textbox(label="Confidence"),
        ],
        title="AI TB Detection + Drug Resistance + Treatment System",
        description=status,
        article=(
            "⚠️ Research/educational prototype — NOT a medical device. "
            "Do not use for diagnosis or treatment decisions."
        ),
        examples=examples,
        theme=gr.themes.Soft(),
    )


def main():
    parser = argparse.ArgumentParser(description="TB X-Ray AI Detector Gradio app")
    parser.add_argument("--host", default="0.0.0.0", help="server host")
    parser.add_argument("--port", type=int, default=12000, help="server port")
    parser.add_argument("--share", action="store_true", help="create a public link")
    args = parser.parse_args()

    build_interface().launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
