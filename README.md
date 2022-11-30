## How to run?
### Training
    python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 256 --data-path YOUR IMAGENET PATH --output_dir '' --vitality