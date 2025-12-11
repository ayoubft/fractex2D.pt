
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) <a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

# Fracture Segmentation on FraXet

This repository contains baseline models for **fracture detection and segmentation** from paired *RGB + DEM* outcrop imagery. The goal is to provide tools using classical **computer vision filters** and deep learning models such as **U‑Net** and **SegFormer** for geological fracture mapping. Models are trained on FraXet ([10.5281/zenodo.17069947](https://doi.org/10.5281/zenodo.17069947)).

## Features

- **Computer vision filters**
- **Model baselines for fracture segmentation** using:
  - U‑Net (CNN encoder–decoder)
  - SegFormer (vision transformer)  
  Both trained to take 4‑channel inputs (RGB + DEM) and produce pixelwise fracture probability maps. ([huggingface.co/ayoubft/fraXteX](https://huggingface.co/ayoubft/fraXteX), [10.5281/zenodo.17866853](https://doi.org/10.5281/zenodo.17866853))
- **Inference tools** for patch‑based prediction on arbitrary imagery.
- **Evaluation scripts** for standard metrics (IoU, accuracy, F1 etc.).
- **Demo** datasets and example scripts for easy use.

---

## Installation

```bash
git clone https://github.com/ayoubft/fractex2D.pt.git
cd fractex2D.pt
pip install -r requirements.txt
````

---

## Inference Scripts

There are several ways to run inference:

1. **Online**
   Try it on [Hugging Face Space](https://huggingface.co/spaces/ayoubft/fractex2D_tuto).

2. **From CLI**

```bash
python infer.py \
  --image path/to/rgb.png \
  --dem path/to/dem.tif \
  --model unet \
  --output pred_mask.png
```

3. **From Python**

```python
from infer_function import run_fracture_inference

mask = run_fracture_inference(
    "rgb.png",
    "dem.tif",
    model_name="segformer",
    output_path="pred.png"
)
mask.show()
```

---

## Training

To train a model:

```bash
# config at config/main.yaml
python train.py
```

---

## Limitations

* Predictions depend on data quality, lighting, and texture conditions.
* Not suitable for safety‑critical use without expert validation.

---

## Citation & Acknowledgements

If you use the FraXet2D baselines in academic work, please cite:

```
Fatihi, A., Caldeira, J., Beucler, T., Thiele, S. T., & Samsu, A. Towards robust fracture mapping: Benchmarking automatic fracture mapping in 2D outcrop imagery. Solid earth. (preprint coming soon)
```