python -m torch.distributed.launch --nproc_per_node=1 train_PBAFN_stage1.py --name PBAFN_stage1 \
--resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 1 --label_nc 14 --launcher pytorch \
--continue_train










