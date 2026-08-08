# Sample X-ray images

This folder holds images you can use to try the Gradio demo quickly.

## What's here

- `sample_normal.png` — a **synthetic** placeholder image (a light-gray gradient
  with noise). It is **not** a real chest X-ray and contains no patient data.
- `sample_tb_like.png` — a **synthetic** placeholder image (a darker gradient
  with noise). Also **not** real medical data.

These exist only so the demo UI can be exercised end-to-end without needing to
download a dataset. They will **not** produce a meaningful TB prediction.

## Getting real (de-identified) X-rays

The model was trained on the Kaggle **TB Chest Radiography Database**
(`tawsifurrahman/tuberculosis-tb-chest-xray-dataset`). To fetch real,
de-identified images for local testing:

```bash
pip install kaggle
# place your kaggle.json in ~/.kaggle/ (chmod 600)
kaggle datasets download -d tawsifurrahman/tuberculosis-tb-chest-xray-dataset
unzip -q tuberculosis-tb-chest-xray-dataset.zip
```

The dataset extracts to `TB_Chest_Radiography_Database/{Normal,Tuberculosis}/`.
Copy a few images into this `samples/` folder (respecting the dataset license)
to test real inference.

> ⚠️ Chest X-rays may be sensitive medical data. Handle them according to
> applicable privacy laws and institutional policies. Do not commit real
> patient images to this repository.
