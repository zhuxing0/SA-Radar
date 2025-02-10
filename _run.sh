conda create --prefix /radar_sim_env python=3.11.9
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python
pip install scikit-image
pip install tensorboard==2.12.0
pip install matplotlib 
pip install tqdm
pip install timm==0.5.4

conda activate /radar_sim_env

# ----------------- implicit simulation ----------------------------#
# eval the pre-train model on RADDet test set
python evaluate.py 

# train and eval your model on RADDet
python train.py --logdir ./checkpoints/icfar_raddet_bs3_lr0.0002_50e --batch_size 3 --train_datasets raddet --segment_mask_loss --lr 0.0002 --epochs 50
python evaluate.py --restore_ckpt ./checkpoints/icfar_raddet_bs3_lr0.0002_50e/icfar-net.pth

# run implicit simulation on RADDet train set by the pre-train model
python demo.py --save_numpy --version train
# run implicit simulation on RADDet test set by the pre-train model
python demo.py --save_numpy --version test

# run implicit simulation on RADDet train set by your model
python demo.py --restore_ckpt ./checkpoints/icfar_raddet_bs3_lr0.0002_50e/icfar-net.pth --save_numpy --version train
# run implicit simulation on RADDet test set by your model
python demo.py --restore_ckpt ./checkpoints/icfar_raddet_bs3_lr0.0002_50e/icfar-net.pth --save_numpy --version test

# ----------------- explicit simulation ----------------------------#
# run explicit simulation on RADDet train set
python explicit_simulation_on_RADDet.py --version train

# run explicit simulation on RADDet test set
python explicit_simulation_on_RADDet.py --version test