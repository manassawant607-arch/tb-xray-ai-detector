"""Generate ai_drp.ipynb: a cleaned, parameterized Colab/Jupyter notebook derived
from ai_drp.py. Run once; the output notebook is committed, not this script."""
import json


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": s}


def lines(*xs):
    return [x + "\n" for x in xs][:-1] if xs else []


cells = [
    md([
        "# AI DRP — TB Detection, Drug Resistance & Treatment (Prototype)\n",
        "\n",
        "Cleaned, parameterized version of `ai_drp.py`.\n",
        "\n",
        "> ⚠️ Research/educational prototype — **NOT** a medical device. "
        "Do not use for diagnosis or treatment decisions.\n",
    ]),
    md(["## 0. Configuration\n"]),
    code([
        "# Tunable parameters — change these instead of editing code below.\n",
        "DATASET_SLUG = \"tawsifurrahman/tuberculosis-tb-chest-xray-dataset\"  # Kaggle dataset\n",
        "DATASET_DIR = \"TB_Chest_Radiography_Database\"\n",
        "IMG_SIZE = (224, 224)\n",
        "BATCH_SIZE = 32\n",
        "EPOCHS_CNN = 10          # from-scratch CNN epochs (was 5)\n",
        "EPOCHS_TRANSFER = 10     # MobileNetV2 transfer-learning epochs (was 5)\n",
        "VALIDATION_SPLIT = 0.2\n",
        "THRESHOLD = 0.5\n",
        "CNN_MODEL_PATH = \"tb_detection_model.h5\"\n",
        "TRANSFER_MODEL_PATH = \"tb_detector_ai.h5\"\n",
    ]),
    md(["## 1. Kaggle setup & dataset download\n",
        "Upload your `kaggle.json` (Kaggle API token) when prompted.\n"]),
    code([
        "!pip install -q kaggle\n",
        "from google.colab import files\n",
        "files.upload()  # upload kaggle.json\n",
        "\n",
        "!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json\n",
        "!kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset\n",
        "!unzip -q tuberculosis-tb-chest-xray-dataset.zip\n",
    ]),
    code([
        "import os\n",
        "for folder in os.listdir(DATASET_DIR):\n",
        "    p = os.path.join(DATASET_DIR, folder)\n",
        "    if os.path.isdir(p):\n",
        "        print(folder, len(os.listdir(p)))\n",
    ]),
    md(["## 2. Preprocessing & augmentation\n",
        "Added light augmentation (flip/rotation) on the training split "
        "for better generalization.\n"]),
    code([
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
        "\n",
        "train_datagen = ImageDataGenerator(\n",
        "    rescale=1./255,\n",
        "    rotation_range=15,\n",
        "    horizontal_flip=True,\n",
        "    validation_split=VALIDATION_SPLIT,\n",
        ")\n",
        "val_datagen = ImageDataGenerator(rescale=1./255, validation_split=VALIDATION_SPLIT)\n",
        "\n",
        "train_data = train_datagen.flow_from_directory(\n",
        "    DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,\n",
        "    class_mode=\"binary\", subset=\"training\")\n",
        "val_data = val_datagen.flow_from_directory(\n",
        "    DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,\n",
        "    class_mode=\"binary\", subset=\"validation\")\n",
    ]),
    md(["## 3a. Model A — from-scratch CNN\n"]),
    code([
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout\n",
        "\n",
        "cnn = Sequential([\n",
        "    Conv2D(32, (3,3), activation=\"relu\", input_shape=(*IMG_SIZE, 3)),\n",
        "    MaxPooling2D(2,2),\n",
        "    Conv2D(64, (3,3), activation=\"relu\"),\n",
        "    MaxPooling2D(2,2),\n",
        "    Conv2D(128, (3,3), activation=\"relu\"),\n",
        "    MaxPooling2D(2,2),\n",
        "    Flatten(),\n",
        "    Dense(128, activation=\"relu\"),\n",
        "    Dropout(0.3),\n",
        "    Dense(1, activation=\"sigmoid\"),\n",
        "])\n",
        "cnn.compile(optimizer=\"adam\", loss=\"binary_crossentropy\", metrics=[\"accuracy\"])\n",
        "cnn.summary()\n",
        "cnn.fit(train_data, validation_data=val_data, epochs=EPOCHS_CNN)\n",
        "cnn.save(CNN_MODEL_PATH)\n",
    ]),
    md(["## 3b. Model B — MobileNetV2 transfer learning (recommended)\n"]),
    code([
        "import tensorflow as tf\n",
        "from tensorflow.keras.applications import MobileNetV2\n",
        "from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout\n",
        "from tensorflow.keras.models import Model\n",
        "\n",
        "base = MobileNetV2(input_shape=(*IMG_SIZE, 3), include_top=False, weights=\"imagenet\")\n",
        "base.trainable = False\n",
        "x = base.output\n",
        "x = GlobalAveragePooling2D()(x)\n",
        "x = Dense(128, activation=\"relu\")(x)\n",
        "x = Dropout(0.3)(x)\n",
        "preds = Dense(1, activation=\"sigmoid\")(x)\n",
        "transfer = Model(inputs=base.input, outputs=preds)\n",
        "transfer.compile(optimizer=\"adam\", loss=\"binary_crossentropy\",\n",
        "                 metrics=[\"accuracy\"])\n",
        "transfer.summary()\n",
        "transfer.fit(train_data, validation_data=val_data, epochs=EPOCHS_TRANSFER)\n",
        "transfer.save(TRANSFER_MODEL_PATH)\n",
    ]),
    md(["## 4. Inference + Gradio demo\n",
        "Loads the trained CNN and exposes a 4-output UI. Run this locally too once "
        "`tb_detection_model.h5` is in the repo root.\n"]),
    code([
        "import gradio as gr\n",
        "import numpy as np\n",
        "from PIL import Image\n",
        "\n",
        "model = tf.keras.models.load_model(CNN_MODEL_PATH)\n",
        "\n",
        "def preprocess(img):\n",
        "    img = img.convert(\"RGB\").resize(IMG_SIZE)\n",
        "    return np.expand_dims(np.array(img)/255.0, axis=0)\n",
        "\n",
        "def predict_tb(image):\n",
        "    pred = float(model.predict(preprocess(image), verbose=0)[0][0])\n",
        "    if pred > THRESHOLD:\n",
        "        return (\"TB Detected\", \"rpoB mutation detected\",\n",
        "                \"Rifampicin Resistant (Possible MDR-TB)\",\n",
        "                \"Bedaquiline + Linezolid + Levofloxacin\")\n",
        "    return (\"Normal\", \"No mutation detected\", \"Drug Sensitive\",\n",
        "            \"Standard TB therapy\")\n",
        "\n",
        "gr.Interface(\n",
        "    fn=predict_tb,\n",
        "    inputs=gr.Image(type=\"pil\", label=\"Upload Chest X-ray\"),\n",
        "    outputs=[gr.Textbox(label=\"TB Detection\"),\n",
        "             gr.Textbox(label=\"Mutation Analysis\"),\n",
        "             gr.Textbox(label=\"Drug Resistance Prediction\"),\n",
        "             gr.Textbox(label=\"Treatment Recommendation\")],\n",
        "    title=\"AI TB Detection + Drug Resistance + Treatment System\",\n",
        ").launch()\n",
    ]),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("ai_drp.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote ai_drp.ipynb with", len(cells), "cells")
