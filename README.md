![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) <a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>


# FraXet2D: Benchmarking Fracture Segmentation

This repository contains baseline models and benchmarking pipelines for **fracture detection and segmentation** from paired *RGB + DEM* outcrop imagery. The goal is to provide **reproducible benchmarks** using classical filtering, convolutional neural networks (UNet), and transformer-based segmentation (SegFormer) for geological fracture mapping.

## Features

- **Computer vision filters**
- **Model baselines for fracture segmentation** using:
  - U‑Net (CNN encoder–decoder)
  - SegFormer (vision transformer)  
  Both trained to take 4‑channel inputs (RGB + DEM) and produce pixelwise fracture probability maps. ([huggingface.co](https://huggingface.co/ayoubft/fraXteX))
- **Benchmark scripts** to compare performance across architectures.
- **Inference tools** for patch‑based prediction on arbitrary imagery.
- Optional evaluation scripts for standard metrics (IoU, accuracy, F1 etc.).
- Demo datasets and scripts for reproducible tests.

---

## Installation

```bash
git clone https://github.com/ayoubft/fractex2D.pt.git
cd fractex2D.pt
pip install -r requirements.txt
````

---

## Inference Scripts

There are two ways to run inference:

1. **Online**
Try it on https://huggingface.co/spaces/ayoubft/fractex2D_tuto.

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
# infere.py file
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

- Predictions depend on data quality, lighting, and texture conditions.
- Not suitable for safety‑critical use without expert validation.

---

## Citation & Acknowledgements

If you use the FraXet2D baselines in research, please cite:

```
@misc{fractex2d,
  author = {Fatihi, Ayoub and others},
  title = {FraXet2D: Benchmarking Fracture Segmentation},
  year = {2025},
  url = {https://github.com/ayoubft/fractex2D.pt}
}
```