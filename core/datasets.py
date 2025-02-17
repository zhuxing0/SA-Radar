import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import random
from pathlib import Path
from glob import glob
import os.path as osp

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor
import pdb
from scipy.ndimage import maximum_filter
import pickle
from tqdm import tqdm
import time

def extract_local_maxima(img):
    output = np.zeros_like(img)
    neighborhood = maximum_filter(img, size=3)
    local_max = (img == neighborhood)
    output[local_max] = img[local_max]
    return output

class zeroDataset(data.Dataset):
    def __init__(self, aug_params=None):
        self.augmentor = None
        if aug_params is not None and "crop_size" in aug_params:
            self.augmentor = FlowAugmentor(**aug_params)

        self.RAD_path_list = []
        self.class_list = ["person", "bicycle", "car", "motorcycle", "bus", "truck" ]
        self.radar_cube_gt_list = []
        self.radar_point_list = []
        self.segment_mask_list = []

    def __getitem__(self, index):
        RAD_path = self.RAD_path_list[index]
        gt_path = self.RAD_path_list[index].replace('/RAD/', '/gt/').replace('.npy', '.pickle')

        real_cube = np.load(RAD_path)
        real_cube = np.abs(real_cube)
        real_cube = pow(real_cube, 2)
        real_cube = np.log10(real_cube + 1.)
        radar_cube_gt = real_cube.transpose((0, 2, 1))

        num_range_bins, num_velocity_bins, num_azimuth_bins = radar_cube_gt.shape
        assert (num_range_bins, num_velocity_bins, num_azimuth_bins) == (256, 64, 256)

        radar_point = np.zeros_like(radar_cube_gt)
        for velocity_i in range(num_velocity_bins):
            radar_point[:,velocity_i,:] = extract_local_maxima(radar_cube_gt[:,velocity_i,:])
        radar_point = radar_point * radar_cube_gt

        segment_mask = np.zeros_like(radar_cube_gt)
        with open(gt_path, "rb") as f:
            radar_instances = pickle.load(f)
            for i in range(len(radar_instances["boxes"])): # "classes"
                class_now = radar_instances["classes"][i]
                xyzwhd = radar_instances["boxes"][i]
                x_begin = int(xyzwhd[0] - xyzwhd[3]//2)
                x_end = int(xyzwhd[0] + xyzwhd[3]//2)
                y_begin = int(xyzwhd[1] - xyzwhd[4]//2)
                y_end = int(xyzwhd[1] + xyzwhd[4]//2)
                z_begin = int(xyzwhd[2] - xyzwhd[5]//2)
                z_end = int(xyzwhd[2] + xyzwhd[5]//2)
                x_begin, x_end = max(0, x_begin), min(num_range_bins, x_end)
                y_begin, y_end = max(0, y_begin), min(num_azimuth_bins, y_end)
                z_begin, z_end = max(0, z_begin), min(num_velocity_bins, z_end)
                segment_mask[x_begin:x_end, z_begin:z_end, y_begin:y_end] = self.class_list.index(class_now) + 1.0
        
        point_mask = radar_point > 0
        segment_mask = segment_mask * point_mask

        radar_cube_gt = torch.from_numpy(radar_cube_gt).float()
        radar_point = torch.from_numpy(radar_point).float().unsqueeze(0)
        segment_mask = torch.from_numpy(segment_mask).float().unsqueeze(0)

        return radar_cube_gt, radar_point, segment_mask
        
    def __len__(self):
        return len(self.RAD_path_list)
    

class RADDet_Dataset(zeroDataset):
    def __init__(self, aug_params=None, root='/gpfs/essfs/iat/Tsinghua/xiaowq/RADDet', split='train'):
        super(RADDet_Dataset, self).__init__(aug_params)

        self.RAD_path_list = sorted(glob(osp.join(root, split, 'RAD/*/*.npy')))

        # RAD_path_list = sorted(glob(osp.join(root, split, 'RAD/*/*.npy')))
        # if split=='train':
        #     for RAD_path in tqdm(RAD_path_list):
        #         gt_path = RAD_path.replace('/RAD/', '/gt/').replace('.npy', '.pickle')
        #         t0 = time.time()
        #         real_cube = np.load(RAD_path)

        #         t1 = time.time()
        #         real_cube = np.abs(real_cube)
        #         t2 = time.time()
        #         real_cube = pow(real_cube, 2)
        #         t3 = time.time()
        #         real_cube = np.log10(real_cube + 1.)
        #         t4 = time.time()
        #         radar_cube_gt = real_cube.transpose((0, 2, 1))
        #         t5 = time.time()

        #         num_range_bins, num_velocity_bins, num_azimuth_bins = radar_cube_gt.shape
        #         if (num_range_bins, num_velocity_bins, num_azimuth_bins) == (256, 64, 256):
        #             self.RAD_path_list.append(RAD_path)
        #         else:
        #             continue
        # else:
        #     self.RAD_path_list = RAD_path_list
        
  
def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    for dataset_name in args.train_datasets:
        if dataset_name == 'raddet':
            new_dataset = RADDet_Dataset(aug_params, root='/gpfs/essfs/iat/Tsinghua/xiaowq/RADDet')
            logging.info(f"Adding {len(new_dataset)} samples from raddet dataset")
        
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=8, drop_last=True) # int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2

    logging.info('Training with %d cubes' % len(train_dataset))
    return train_loader

