
python train.py --logdir ./checkpoints_icfar/icfar_raddet_lr0.0002_50e --batch_size 3 --train_datasets raddet --hourglass_type 2 --segment_mask_loss --lr 0.0002 --epochs 50

python demo_icfar.py --restore_ckpt ./checkpoints_icfar/icfar_raddet_bs3_hourglassv2_smloss_lr0.0002_50e/icfar-net.pth --hourglass_type 2
python demo_icfar.py --restore_ckpt ./checkpoints_icfar/icfar_raddet_bs3_hourglassv2_smloss_lr0.0002_50e/icfar-net.pth --hourglass_type 2 --save_numpy
python demo_icfar.py --restore_ckpt ./checkpoints_icfar/icfar_raddet_bs3_hourglassv2_smloss_lr0.0002_50e/icfar-net.pth --hourglass_type 2 --save_numpy --radar_path /gpfs/essfs/iat/Tsinghua/xiaowq/RADDet/train --save_dir /gpfs/essfs/iat/Tsinghua/xiaowq/RADDet/train_sim
