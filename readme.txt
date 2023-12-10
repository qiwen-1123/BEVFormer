# Run the docker container
docker run -it --userns=host --network=host --gpus all --mount type=bind,src="$(pwd)/",target="/BevFormer/" --shm-size 32G bevformer-cuda11.1

# Setup .vscode (settings.json)
{
    "python.defaultInterpreterPath": "/root/miniconda3/envs/bin/python"
}

# run test
python -m torch.distributed.launch --nproc_per_node=1 --master_port=-29503 \
    $(dirname "$0")/test.py ./projects/configs/bevformer/bevformer_base.py ./ckpts/bevformer_r101_dcn_24ep.pth --launcher pytorch ${@:4} --eval bbox

