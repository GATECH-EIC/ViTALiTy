python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 16 --output_dir test --vitality
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 256 --output_dir TIny_xnorm --resume 'TIny_xnorm/checkpoint.pth'
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 256 --output_dir Tiny_test_pre --resume https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --lr 1e-4 --epochs 300 --batch-size 128 --output_dir Small_DwS --resume 'Small_DwS/checkpoint.pth'
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_small_patch16_224 --lr 1e-4 --epochs 300 --batch-size 128 --output_dir Small_Sparse --resume 'Small_Sparse/checkpoint.pth'

# python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_224 --lr 1e-4 --epochs 300 --batch-size 64 --output_dir Base_DwS --resume 'Base_DwS/checkpoint.pth'
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_base_patch16_224 --lr 1e-4 --epochs 300 --batch-size 64 --output_dir Base_Sparse --resume 'Base_Sparse/checkpoint.pth'

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 50 --batch-size 256 --output_dir test2
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 256 --output_dir Tiny_DwS_lr1 --resume 'Tiny_DwS_lr1/checkpoint.pth'
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 5e-4 --epochs 300 --batch-size 256 --output_dir Tiny_DwS_noQ --resume 'Tiny_DwS_noQ/checkpoint.pth'
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 256 --output_dir Test --resume 'Tiny_DwS_Pre/checkpoint.pth'

#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 5e-4 --epochs 300 --batch-size 256 --output_dir Tiny_normV --resume 'Tiny_normV/checkpoint.pth'
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 5e-4 --epochs 300 --batch-size 256 --output_dir Tiny_normAll --resume 'Tiny_normAll/checkpoint.pth'
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 256 --output_dir Tiny_normAll_lr1 --resume 'Tiny_normAll_lr1/checkpoint.pth'
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 256 --output_dir Tiny_Sparse2 --resume 'Tiny_Sparse2/checkpoint.pth' 
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 256 --output_dir Tiny_Sparse --resume 'Tiny_Sparse/best_checkpoint.pth' 
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 5e-4 --epochs 300 --batch-size 256 --output_dir Tiny_Dw2 --resume 'Tiny_Dw2/checkpoint.pth'
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 5e-4 --epochs 300 --batch-size 256 --output_dir Tiny_Dw --resume 'Tiny_Dw/checkpoint.pth'
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 1e-3 --epochs 300 --batch-size 256 --output_dir Tiny_lr --resume 'Tiny_lr/checkpoint.pth'
#python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model deit_tiny_patch16_224 --lr 5e-4 --epochs 500 --batch-size 256 --output_dir Tiny_Sparse_pre --resume 'Tiny_Sparse_pre/checkpoint.pth'
