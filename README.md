# WNet2D: Multi-Scale and State-Space Enhanced Network for Medical Image Segmentation
This repository provides a minimal, reproducible implementation of WNet2D:

Scope. Train/evaluate on Kvasir-SEG (with templates for DRIVE/ISIC2017/PH2), reproduce Table 2 (Dice/Jaccard/95HD/ASD), and reproduce Table 3/5 efficiency numbers under a unified protocol.

What’s included. train.py, eval.py, dataset configs (kvasir.yaml, drive.yaml, isic2017.yaml, ph2.yaml), and a unified measurement utility measure.py.

Environments. Pinned via environment.yml (Conda, CUDA 11.8) and requirements.txt.

Versioning. Camera-ready tag: v1.0-camera-ready (commit 81a39a8).

> Camera-ready tag: **v1.0-camera-ready**
<p align="center">
  <img src="arch.png" width="720" alt="WNet2D Architecture">
</p>
## 1. Environment
```bash
# Conda (CUDA 11.8)
conda env create -f environment.yml
conda activate wnet2d

# or pip
pip install -r requirements.txt
```

## 2. Data
Expected layout (Kvasir-SEG):
```
data/
  Kvasir-SEG/
    images/    # *.jpg/*.png
    masks/     # binary masks (0/255)
```
You can customize paths via YAML configs in `configs/`.

## 3. Train (example: Kvasir-SEG)
```bash
python train.py --cfg kvasir.yaml
```

## 4. Evaluate & export metrics (Dice/IoU/95HD/ASD)
```bash
python eval.py --cfg kvasir.yaml --ckpt checkpoints/wnet2d_kvasir.pth
```
Metrics are reported per-image and summarized into a CSV. **Distances are computed in pixels**.

## 5. Efficiency measurement (protocol identical to the paper)
```bash
# default: reserved (nvidia-smi "Used")
python measure.py --model wnet2d --size 512 --precision fp32 --runs 200 --warmup 50 --report reserved

# also possible (for reference only):
python -m wnet2d.utils.measure --report allocated
```
**Protocol.** Latency is the mean per-image inference time (batch size = 1, input = 512×512) over 200 runs after 50 warm-up runs, measured with CUDA events + `cudaDeviceSynchronize()` under **PyTorch 2.1.2 (CUDA 11.8), FP32**. **GPU memory refers to the maximum process memory ("Used") reported by `nvidia-smi`** (equivalently, PyTorch **peak reserved** via `torch.cuda.max_memory_reserved()`). Throughput is **FPS = 1000 / Latency (ms)**.

## 6. Reproduce tables
- **Table 2** (Kvasir-SEG): `python -m wnet2d.engine.eval --cfg configs/kvasir.yaml --ckpt checkpoints/wnet2d_kvasir.pth`
- **Table 3** (4090): `python -m wnet2d.utils.measure --model wnet2d --size 512 --precision fp32 --runs 200 --warmup 50 --report reserved`
- **Table 5** (Mamba depth sensitivity): use `configs/kvasir_depth{2,3,4}.yaml` then run the same measure command.

## 7. Citation
See `CITATION.cff` or cite the paper once available.
