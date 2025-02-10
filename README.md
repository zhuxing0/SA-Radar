# RadarSimReal

RadarSimReal is a project for simulating radar data, supporting both implicit and explicit simulations. This document provides guidance on installation, running, and using the project.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Train and Evaluate Model](#train-and-evaluate-model)
- [Run Implicit Simulation](#run-implicit-simulation)
- [Run Explicit Simulation](#run-explicit-simulation)
- [Notes](#notes)

## Environment Setup

First, create a new Conda environment and specify the Python version:

```bash
conda create --prefix /radar_sim_env python=3.11.9
conda activate /
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python
pip install scikit-image
pip install tensorboard==2.12.0
pip install matplotlib
pip install tqdm
pip install timm==0.5.4
```

## train-and-evaluate-model for implicit-simulation

### eval the pre-train model on RADDet test set
```python
python evaluate.py 
```

### train and eval your model on RADDet
```python
python train.py --logdir ./checkpoints/icfar_raddet_bs3_lr0.0002_50e --batch_size 3 --train_datasets raddet --segment_mask_loss --lr 0.0002 --epochs 50
python evaluate.py --restore_ckpt ./checkpoints/icfar_raddet_bs3_lr0.0002_50e/icfar-net.pth
```

## run-implicit-simulation

### run implicit simulation on RADDet train set by the pre-train model
```python
python demo.py --save_numpy --version train
```

### run implicit simulation on RADDet test set by the pre-train model
```python
python demo.py --save_numpy --version test
```

### run implicit simulation on RADDet train set by your model
```python
python demo.py --restore_ckpt ./checkpoints/icfar_raddet_bs3_lr0.0002_50e/icfar-net.pth --save_numpy --version train
```

### run implicit simulation on RADDet test set by your model
```python
python demo.py --restore_ckpt ./checkpoints/icfar_raddet_bs3_lr0.0002_50e/icfar-net.pth --save_numpy --version test
```

## run-explicit-simulation

### run explicit simulation on RADDet train set
```python
python explicit_simulation_on_RADDet.py --version train
```

### run explicit simulation on RADDet test set
```python
python explicit_simulation_on_RADDet.py --version test
```

## notes