export CUDA_PATH=/usr/local/cuda-10.0
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
python test.py --name demo --resize_or_crop None --batchSize 1 --gpu_ids 0
