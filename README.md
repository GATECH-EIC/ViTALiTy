
#ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](https://opensource.org/licenses/Apache-2.0)

Jyotikrishna Dass*, Shang Wu*, Huihong Shi*, Chaojian Li, Zhifan Ye, Zhongfeng Wang and Yingyan Lin
(*Equal contribution)

Accepted by [HPCA 2023](https://hpca-conf.org/2023/). More Info:
\[ [**Paper**](https://arxiv.org/abs/2211.05109) | [**Slide**]() | [**Github**](https://github.com/GATECH-EIC/ViTaLiTy) \]

---

## Overview of the Co-Design Framework

We propose a low-rank and sparse approximation algorithm and accelerator Co-Design framework dubbed ViTALiTy.

* ***On the algorithm level***, 

<p align="center">
<img src="./figures/Algorithm.png" width="800">
</p>

* ***On the hardware level***, 

<p align="center">
<img src="./figures/Hardware.png" width="800">
</p>

## How to run?
### Training
    python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 256 --data-path YOUR IMAGENET PATH --output_dir '' --vitality