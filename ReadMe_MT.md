# Run the docker container
    docker run -it --userns=host --network=host --gpus all --mount type=bind,src="$(pwd)/",target="/BevFormer/" --shm-size 32G bevformer-cuda11.1

# Setup .vscode (settings.json)
    {
        "python.defaultInterpreterPath": "/root/miniconda3/envs/bin/python"
    }

# run test
    python -m torch.distributed.launch --nproc_per_node=1 --master_port=-29503 \
        $(dirname "$0")/test.py ./projects/configs/bevformer/bevformer_base.py ./ckpts/bevformer_r101_dcn_24ep.pth --launcher pytorch ${@:4} --eval bbox

# Code Modification
## new Detection Head
1. new .py file `projects/mmdet3d_plugin/bevformer/dense_heads/bevformer_head_2D.py` for projecting 3D BBox into 2D image
2. new .py file `projects/mmdet3d_plugin/bevformer/detectors/bevformer_2D.py` modify original bevformer with 2D Head
3. new config file `projects/configs/bevformer/bevformer_small_2D.py` config file for 2D version