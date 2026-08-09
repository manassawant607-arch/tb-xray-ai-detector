"""Set up Kaggle credentials and download the real datasets used by ai_drp.py.

This script reproduces the Colab setup steps from ai_drp.py locally:

    !pip install kaggle                       # ai_drp.py line 10
    !mkdir -p ~/.kaggle                       # line 15
    !cp kaggle.json ~/.kaggle/                # line 16
    !chmod 600 ~/.kaggle/kaggle.json          # line 17
    !kaggle datasets download tawsifurrahman/tuberculosis-tb-chest-xray-dataset  # line 21
    !unzip -q tuberculosis-tb-chest-xray-dataset.zip                            # line 19
    !kaggle datasets download -d khushikyad001/tuberculosis-trends              # line 135
    !unzip tuberculosis-trends-global-and-regional-insights.zip                 # line 137

Place your kaggle.json (from https://www.kaggle.com/settings → API → Create New Token)
in the repo root, then run:

    python setup_kaggle.py

It installs kaggle, copies kaggle.json to ~/.kaggle with safe permissions, and
downloads + unzips both datasets used by the full ai_drp.py pipeline.
"""

import os
import shutil
import stat
import subprocess
import sys

KAGGLE_DIR = os.path.expanduser("~/.kaggle")
REPO_KAGGLE_JSON = "kaggle.json"

TB_DATASET = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"
TB_ZIP = "tuberculosis-tb-chest-xray-dataset.zip"
TB_DIR = "TB_Chest_Radiography_Database"

TRENDS_DATASET = "khushikyad001/tuberculosis-trends-global-and-regional-insights"
TRENDS_ZIP = "tuberculosis-trends-global-and-regional-insights.zip"
TRENDS_CSV = "Tuberculosis_Trends.csv"


def run(cmd):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def install_kaggle():
    """ai_drp.py line 10: !pip install kaggle"""
    run([sys.executable, "-m", "pip", "install", "--quiet", "kaggle"])


def setup_credentials(src=REPO_KAGGLE_JSON):
    """ai_drp.py lines 15-17: mkdir ~/.kaggle, cp kaggle.json, chmod 600."""
    if not os.path.exists(src):
        print(f"ERROR: {src} not found.")
        print("Download it from https://www.kaggle.com/settings -> API -> Create New Token")
        return False
    os.makedirs(KAGGLE_DIR, exist_ok=True)
    dest = os.path.join(KAGGLE_DIR, "kaggle.json")
    shutil.copy(src, dest)
    os.chmod(dest, stat.S_IRUSR | stat.S_IWUSR)  # 600
    print(f"credentials -> {dest} (chmod 600)")
    return True


def download_tb_dataset():
    """ai_drp.py lines 19, 21: download + unzip the TB chest X-ray dataset."""
    if os.path.isdir(TB_DIR):
        n = sum(len(os.listdir(os.path.join(TB_DIR, d)))
                for d in ("Normal", "Tuberculosis") if os.path.isdir(os.path.join(TB_DIR, d)))
        print(f"[skip] {TB_DIR} already present ({n} images)")
        return
    run(["kaggle", "datasets", "download", "-d", TB_DATASET])
    run(["unzip", "-q", TB_ZIP])


def download_trends():
    """ai_drp.py lines 135, 137: download + unzip the TB trends CSV."""
    if os.path.exists(TRENDS_CSV):
        print(f"[skip] {TRENDS_CSV} already present")
        return
    run(["kaggle", "datasets", "download", "-d", TRENDS_DATASET])
    run(["unzip", "-q", TRENDS_ZIP])


def main():
    try:
        install_kaggle()
    except subprocess.CalledProcessError:
        print("WARNING: could not install kaggle package")
    if not setup_credentials():
        return 1
    download_tb_dataset()
    download_trends()
    print("\nDatasets ready. Now run the full ai_drp.py-equivalent pipeline:")
    print("  python train_real_model.py    # CNN + MobileNetV2 + RandomForest")
    print("  python evaluate_model.py      # metrics + confusion matrix")
    print("  python app.py                  # Gradio app with X-ray validity gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
