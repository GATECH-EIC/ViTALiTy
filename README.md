
# ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](https://opensource.org/licenses/Apache-2.0)

Jyotikrishna Dass*, Shang Wu*, Huihong Shi*, Chaojian Li, Zhifan Ye, Zhongfeng Wang and Yingyan Lin

(*Equal contribution)

Accepted by [HPCA 2023](https://hpca-conf.org/2023/). More Info:
\[ [**Paper**](https://arxiv.org/abs/2211.05109) | [**Slide**]() | [**GitHub**](https://github.com/GATECH-EIC/ViTaLiTy) \]

---

## Overview of the ViTALiTy Framework

We propose a low-rank and sparse approximation algorithm and accelerator co-design framework dubbed ViTALiTy.

* ***On the algorithm level***, we propose a linear attention for reducing the computational and memory cost by decoupling the vanilla softmax attention into its corresponding “weak” and “strong” Taylor attention maps. Unlike the vanilla attentions, the linear attention in ViTALiTy generates a global context matrix G by multiplying the keys with the values. Then, we unify the low-rank property of the linear attention with a sparse approximation of “strong” attention for training the ViT model. Here, the low-rank component of our ViTALiTy attention captures global information with a linear complexity, while the sparse component boosts the accuracy of linear attention model by enhancing its local feature extraction capacity.

<p align="center">
<img src="./figures/ViTALiTY-workflow.png" width="800">
</p>
<p align = "center">
Fig.1 - ViTALiTy workflow comprising the proposed (Low-Rank) Linear Taylor attention (order, m = 1): (i) Higher-order Taylor terms (m > 1)
when added results in vanilla softmax attention score, (ii) Training phase (unifying low-rank and sparse approximation) where higher-order Taylor terms are approximated as Sparse attention (computed using SANGER [28]), and (iii) Inference phase that uses only the (Low-Rank) Linear Taylor attention.
</p>


<p align="center">
<img src="./figures/TaylorAttentionFlow2.png" width="400">
</p>
<p align = "center">
Fig.2 - Computational steps (a) vanilla Softmax Attention and (b) our Taylor attention (see Algorithm 1), where the global context matrix G provides linear computation and memory benefits over the vanilla quadratic QK^T.
</p>

* ***On the hardware level***, we develop a dedicated accelerator to better leverage the algorithmic properties of ViTALiTy’s linear attention, where only a low-rank component is executed during inference favoring hardware efficiency. Specifically, ViTALiTy's accelerator features a chunk-based design integrating both a systolic array tailored for matrix multiplications and pre/post-processors customized for ViTALiTy attentions’ pre/post-processing steps. Furthermore, we adopt an intra-layer pipeline design to leverage the intra-layer data dependency for enhancing the overall throughput together with a down-forward accumulation dataflow for the systolic array to improve hardware efficiency.

<p align="center">
<img src="./figures/hardware_overall.png" width="800">
</p>
<p align = "center">
Fig.3 - An illustration of our ViTALiTy accelerator, which adopts four memory hierarchies (i.e., DRAM, SRAM, NoC, and Regs) to enhance data locality and multiple chunks/sub-processors consisting of a few pre/post-processors and a systolic array to accelerate dedicated operations. Specifically, the pre-processors include an accumulator array for performing column(token)-wise summation, and a divider array and a adder array for conducting element-wise divisions and additions, respectively; In addition, the systolic array (SA) is partitioned into a smaller sub-array named SA-Diag to compute the matrix and diagonal matrix multiplications considering their smaller number of multiplications, and a larger sub-array dubbed SA-General to process the remaining matrix multiplications.
</p>

## How to run?
### Environment set up

    pip install -r requirment.txt
### Training (DeiT-Tiny with vanilla softmax)
    cd src
    python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 256 --data-path YOUR IMAGENET PATH --output_dir ''
### Training (DeiT-Tiny with ViTALiTy)
    cd src
    python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 256 --data-path YOUR IMAGENET PATH --output_dir '' --vitality
### Inference (DeiT-Tiny with vanilla softmax)
    cd src
    python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --batch-size 256 --data-path YOUR IMAGENET PATH --output_dir '' --eval
### Inference (DeiT-Tiny with ViTALiTy)
    cd src
    python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --batch-size 256 --data-path YOUR IMAGENET PATH --output_dir '' --vitality --eval

## Acknowledgment
This codebase is inspired from https://github.com/facebookresearch/deit
