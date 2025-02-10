# RadarSimReal

RadarSimReal is a project for simulating radar data, supporting both implicit and explicit simulations. This document provides guidance on installation, running, and using the project.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset Preparation](#Dataset-Preparation)
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

## Dataset Preparation

Please keep the same directory tree as shown in [GoogleDrive](https://drive.google.com/drive/u/1/folders/1v-AF873jP8p6waChF3pSSqz6HXOOZgkC) or [OneDrive](https://uottawa-my.sharepoint.com/personal/azhan085_uottawa_ca/_layouts/15/guestaccess.aspx?folderid=016d9f28644214b8c8813d618a3809365&authkey=ARvXPjc---r2wTFL6TEuY84&e=bwnfSO)

Download the dataset and arrange it as the folloing directory tree,
```bash
|-- train
	|-- RAD
		|-- part1
			|-- ******.npy
			|-- ******.npy
		|-- part2
			|-- ******.npy
			|-- ******.npy
	|-- gt
		|-- part1
			|-- ******.pickle
			|-- ******.pickle
		|-- part2
			|-- ******.pickle
			|-- ******.pickle
	|-- stereo_image
		|-- part1
			|-- ******.jpg
			|-- ******.jpg
		|-- part2
			|-- ******.jpg
			|-- ******.jpg
|-- test
	|-- RAD
		|-- part1
            |-- ***
	|-- gt
		|-- part1
            |-- ***
	|-- ***
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