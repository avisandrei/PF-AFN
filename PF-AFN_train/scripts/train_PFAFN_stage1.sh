python -m torch.distributed.launch --nproc_per_node=1 --master_port=4703 train_PFAFN_stage1.py --name PFAFN_stage1  \
--PBAFN_warp_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_warp_epoch_006.pth' --PBAFN_gen_checkpoint 'checkpoints/PBAFN_e2e/PBAFN_gen_epoch_006.pth'  \
--lr 0.00003 --niter 100 --niter_decay 100 --resize_or_crop None --verbose --tf_log --batchSize 2 --num_gpus 1 --label_nc 14 --launcher pytorch \
--continue_train










