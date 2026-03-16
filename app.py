import gradio as gr

def detect_tb(image):
    return "TB Detection Model Running"

interface = gr.Interface(
    fn=detect_tb,
    inputs=gr.Image(),
    outputs="text",
    title="AI TB X-Ray Detector"
)

interface.launch()
