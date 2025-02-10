
python train.py --logdir ./checkpoints/icfar_raddet_bs3_lr0.0002_50e --batch_size 3 --train_datasets raddet --segment_mask_loss --lr 0.0002 --epochs 50

python evaluate.py 

python demo.py --restore_ckpt ./checkpoints/icfar_raddet_bs3_lr0.0002_50e/icfar-net.pth
python demo.py --restore_ckpt ./checkpoints/icfar_raddet_bs3_lr0.0002_50e/icfar-net.pth --save_numpy
python demo.py --restore_ckpt ./checkpoints/icfar_raddet_bs3_lr0.0002_50e/icfar-net.pth --save_numpy --radar_path /gpfs/essfs/iat/Tsinghua/xiaowq/RADDet/train --save_dir RADDet_train_sim
