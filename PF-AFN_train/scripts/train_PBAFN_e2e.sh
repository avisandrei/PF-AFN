python -m torch.distributed.launch --nproc_per_node=1 --master_port=4736 train_PBAFN_e2e.py --name PBAFN_e2e   \
--PBAFN_warp_checkpoint 'checkpoints/PBAFN_stage1/PBAFN_warp_epoch_006.pth' --resize_or_crop None --verbose --tf_log --batchSize 4 --num_gpus 1 --label_nc 14 --launcher pytorch \
--continue_train










