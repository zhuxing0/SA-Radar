# Simulate Any Radar: Attribute-Controllable Radar Simulation via Waveform Parameter Embedding

[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://zhuxing0.github.io/projects/SA-Radar)
[![Website](assets/badge-website.svg)](https://zhuxing0.github.io/projects/SA-Radar)

This repo contains the official code of our paper: [Simulate Any Radar: Attribute-Controllable Radar Simulation via Waveform Parameter Embedding](https://zhuxing0.github.io/projects/SA-Radar).

## News

- June 4, 2025: 🚀 Release paper and project page.

- June 4, 2025: 🚀 Release paper and project page.

---

<p align="center">
  <img src="assets/radartop_00.png" alt="Overview" width="100%"/>
</p>

(a) SA-Radar enables controllable and realistic radar simulation by conditioning on
customizable radar attributes. It supports flexible scene editing such as attribute modification, actor
removal, and novel trajectories. (b) SA-Radar improves performance on various tasks including
semantic segmentation, 2D/3D object detection. In all settings, SA-Radar’s synthetic data either
matches or surpasses real data, and provides consistent gains when combined with real-world datasets.

## Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset Preparation](#Dataset-Preparation)
- [Evaluate the Pre-train Model](#evaluate-the-pre-train-model)
- [Train Your Model](#train-your-model)
- [Run Radar Simulation](#run-radar-simulation)
- [Notes](#notes)

## Environment Setup

First, create a new Conda environment and specify the Python version:

```bash
conda create -n radar_sim_env python=3.11.9
conda activate radar_sim_env
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python
pip install scikit-image
pip install tensorboard==2.12.0
pip install matplotlib
pip install tqdm
pip install timm==0.5.4
pip install numpy==1.26.4
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

## Evaluate the pre-train model
```python
python evaluate.py --restore_ckpt ./models/icfar-net.pth --attribute
```

## Train your model

### prepare for the mixed dataset
```python
python make_mixed_dataset_step1.py 
python make_mixed_dataset_step2.py 
```

### train and eval your model on the mixed dataset
```python
python train.py --logdir ./checkpoints/icfar_mixed_bs3_lr0.0002_50e --train_datasets raddet carrada raddet_by_mr --attribute --segment_mask_loss --l1_loss --sml1_loss
python evaluate.py --restore_ckpt ./checkpoints/icfar_mixed_bs3_lr0.0002_50e/icfar-net.pth --attribute
```

## Run-radar-simulation

### run radar simulation on RADDet train set
```python
python demo.py --restore_ckpt ./models/icfar-net.pth --save_numpy --version train --attribute
or 
python demo.py --restore_ckpt ./checkpoints/icfar_mixed_bs3_lr0.0002_50e/icfar-net.pth --save_numpy --version train --attribute
```

### Scene Editing

#### Attribute Modification

Modify the *attribute_list* in demo.py directly.

#### Novel Trajectories
```python
python demo.py --restore_ckpt ./models/icfar-net.pth --save_numpy --version train --attribute --angle_rotation
```

#### Actor Removal
```python
python demo.py --restore_ckpt ./models/icfar-net.pth --save_numpy --version train --attribute --remove
```

## notes
