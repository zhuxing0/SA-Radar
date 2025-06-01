import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
from glob import glob
import os.path as osp
import json
from scipy.ndimage import maximum_filter
import pickle

def compute_the_Attribute(sigma_r_bias, N_list_bias, num_log_a_bias, g_bias=1.0):
    sigma = 2.6 + sigma_r_bias
    g = g_bias * 3
    pangban = 0.1
    
    N_list = [int(8+N_list_bias), int(9+N_list_bias)]
    num_log_a_list = [int(2+num_log_a_bias), int(3+num_log_a_bias)]
    a_bias_list = [1, 1.1, 1.2, 1.3, 1.4, 1.5]

    Rs_list, lambda_list = [], []
    num_azimuth_bins = 256
    azimuth_i = 128
    for N_now in N_list:
        for num_log_a in num_log_a_list:
            for a_bias in a_bias_list:
                window = np.arange(0, N_now).astype(np.float64)
                window = (1-pangban)-pangban*np.cos(2*np.pi*window/N_now-1)
                spectrum = np.fft.fft(window, n=num_azimuth_bins)
                spectrum = np.fft.fftshift(spectrum)
                PSF_a = np.abs(np.roll(spectrum, azimuth_i-num_azimuth_bins//2)) 
                for _ in range(num_log_a):
                    PSF_a = 10 * np.log10(PSF_a + a_bias)

                neighborhood = maximum_filter(PSF_a, size=3)
                local_max = (np.where(PSF_a == neighborhood))[0]
                num_max = len(local_max)

                assert local_max[(num_max-1)//2] == 128
                Rs_list.append(local_max[(num_max-1)//2+1] - local_max[(num_max-1)//2-1])
                lambda_list.append(PSF_a[local_max[(num_max-1)//2+1]] / PSF_a[local_max[(num_max-1)//2]])
    
    Rs = np.mean(Rs_list) / 87.0
    lambda_ = np.mean(lambda_list)

    return sigma, g, Rs, lambda_

def extract_local_maxima(img):
    output = np.zeros_like(img)
    neighborhood = maximum_filter(img, size=3)
    local_max = (img == neighborhood)
    output[local_max] = img[local_max]
    return output

class zeroDataset(data.Dataset):
    def __init__(self, aug_params=None, dataset='raddet', Attribute=False, real_data=True):
        self.augmentor = None

        self.RAD_path_list = []
        self.class_list = ["person", "bicycle", "car", "motorcycle", "bus", "truck" ]
        self.radar_cube_gt_list = []
        self.radar_point_list = []
        self.segment_mask_list = []
        self.dataset=dataset
        self.box_list = []

        self.Attribute = Attribute
        self.Attribute_list = []
        self.real_data = real_data

    def __getitem__(self, index):
        if self.dataset == 'raddet':
            RAD_path = self.RAD_path_list[index]
            gt_path = self.RAD_path_list[index].replace('/RAD/', '/gt/').replace('.npy', '.pickle')

            real_cube = np.load(RAD_path)
            if self.real_data:
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

            if self.Attribute:
                sigma, g, Rs, lambda_ = self.Attribute_list[index]
                radar_cube_zero = torch.zeros_like(radar_point)
                sigma, g, Rs, lambda_ = radar_cube_zero+sigma, radar_cube_zero+g, radar_cube_zero+Rs, radar_cube_zero+lambda_
                return radar_cube_gt, radar_point, segment_mask, sigma, g, Rs, lambda_
            else:
                return radar_cube_gt, radar_point, segment_mask
        elif self.dataset == 'carrada':
            RAD_path = self.RAD_path_list[index]
            real_cube = np.load(RAD_path)
            if self.real_data:
                real_cube = pow(real_cube, 2)
                real_cube = np.log10(real_cube + 1.)
            radar_cube_gt = real_cube.transpose((0, 2, 1)) # 256, 64, 256

            num_range_bins, num_velocity_bins, num_azimuth_bins = radar_cube_gt.shape
            assert (num_range_bins, num_velocity_bins, num_azimuth_bins) == (256, 64, 256)

            radar_point = np.zeros_like(radar_cube_gt)
            for velocity_i in range(num_velocity_bins):
                radar_point[:,velocity_i,:] = extract_local_maxima(radar_cube_gt[:,velocity_i,:])
            radar_point = radar_point * radar_cube_gt

            segment_mask = np.zeros_like(radar_cube_gt)
            box_list = self.box_list[index]
            for box in box_list:
                [x_begin, x_end, y_begin, y_end, z_begin, z_end, label] = box

                segment_mask[x_begin:x_end, z_begin:z_end, y_begin:y_end] = label + 1.0

            point_mask = radar_point > 0
            segment_mask = segment_mask * point_mask

            radar_cube_gt = torch.from_numpy(radar_cube_gt).float()
            radar_point = torch.from_numpy(radar_point).float().unsqueeze(0)
            segment_mask = torch.from_numpy(segment_mask).float().unsqueeze(0)

            if self.Attribute:
                sigma, g, Rs, lambda_ = self.Attribute_list[index]
                radar_cube_zero = torch.zeros_like(radar_point)
                sigma, g, Rs, lambda_ = radar_cube_zero+sigma, radar_cube_zero+g, radar_cube_zero+Rs, radar_cube_zero+lambda_
                return radar_cube_gt, radar_point, segment_mask, sigma, g, Rs, lambda_
            else:
                return radar_cube_gt, radar_point, segment_mask
        
    def __len__(self):
        return len(self.RAD_path_list)
    
class RADDet_Dataset(zeroDataset):
    def __init__(self, aug_params=None, root='dataset/RADDet', split='train', attribute=False, Attributes=None, real_data=True):
        super(RADDet_Dataset, self).__init__(aug_params, Attribute=attribute, real_data=real_data)
        RAD_path_list = sorted(glob(osp.join(root, split, 'RAD/*/*.npy'))) + sorted(glob(osp.join(root, split, 'RAD/*.npy')))
        
        for RAD_path in RAD_path_list:
            self.RAD_path_list.append(RAD_path)
            if attribute:
                if Attributes is not None:
                    self.Attribute_list.append(Attributes)
                else:
                    RAD_name = os.path.basename(RAD_path).replace('.npy', '')
                    sigma_r_bias, N_list_bias, num_log_a_bias, g_bias = float(RAD_name.split('_')[-4]), float(RAD_name.split('_')[-3]), float(RAD_name.split('_')[-2]), float(RAD_name.split('_')[-1])
                    self.Attribute_list.append(compute_the_Attribute(sigma_r_bias, N_list_bias, num_log_a_bias, g_bias))

class Carrada_Dataset(zeroDataset):
    def __init__(self, aug_params=None, root='dataset/Carrada', split='train', attribute=False, Attributes=None, real_data=True):
        super(Carrada_Dataset, self).__init__(aug_params, 'carrada', Attribute=attribute, real_data=real_data)
        split_json_path = osp.join('dataset/Carrada', 'Carrada/data_seq_ref.json')
        with open(split_json_path, 'r') as f:
            split_json = json.load(f)

        if split == 'train':
            split_list = ["Train", "Validation"]
        else:
            split_list = ["Test"]

        RAD_path_list = []
        for date_name, v in split_json.items():
            if v["split"] in split_list:
                RAD_path_list += sorted(glob(osp.join(root, f'datasets_master/*/{date_name}/*/*.npy')))

        self.RAD_path_list = []
        self.box_list = []
        for RAD_path in RAD_path_list:
            boxes = []
            date_name = RAD_path.split('/')[-3]
            rad_name = RAD_path.split('/')[-1].split('.')[0]
            label_ra_path = osp.join('dataset/Carrada', f'Carrada/{date_name}/annotations/box/range_angle_light.json')
            label_rd_path = osp.join('dataset/Carrada', f'Carrada/{date_name}/annotations/box/range_doppler_light.json')
            with open(label_ra_path, 'r') as f:
                label_ra = json.load(f)
                if rad_name not in label_ra.keys():
                    continue
                boxes_ra = label_ra[rad_name]["boxes"]
            with open(label_rd_path, 'r') as f:
                label_rd = json.load(f)
                if rad_name not in label_rd.keys():
                    continue
                boxes_rd = label_rd[rad_name]["boxes"]
            labels = label_ra[rad_name]["labels"]

            if len(boxes_ra) == 0 or len(boxes_ra) != len(boxes_rd):
                continue

            for box_ra, box_rd, label in zip(boxes_ra, boxes_rd, labels):
                x_begin, x_end, y_begin, y_end, z_begin, z_end = box_ra[0], box_ra[2], box_ra[1], box_ra[3], box_rd[1], box_rd[3]
                box = [x_begin, x_end, y_begin, y_end, z_begin, z_end, label]
                boxes.append(box)

            self.RAD_path_list.append(RAD_path)
            self.box_list.append(boxes)
            if attribute:
                self.Attribute_list.append(Attributes)

def fetch_dataloader(args):
    """ Create the data loader """

    aug_params = None

    train_dataset = None
    for dataset_name in args.train_datasets:
        if dataset_name == 'raddet':
            new_dataset = RADDet_Dataset(aug_params, root='dataset/RADDet', attribute=args.attribute, Attributes=compute_the_Attribute(sigma_r_bias=0.057, N_list_bias=0, num_log_a_bias=0))
            logging.info(f"Adding {len(new_dataset)} samples from raddet dataset")

        elif dataset_name == 'carrada':
            new_dataset = Carrada_Dataset(aug_params, root='dataset/Carrada', attribute=args.attribute, Attributes=compute_the_Attribute(sigma_r_bias=0.2, N_list_bias=0, num_log_a_bias=0))
            logging.info(f"Adding {len(new_dataset)} samples from carrada dataset")

        elif dataset_name == 'raddet_by_mr':
            new_dataset = RADDet_Dataset(aug_params, root='sim_output/EXsim_onRADDAT_multi-radar', split='train', attribute=args.attribute, Attributes=None, real_data=False)
            logging.info(f"Adding {len(new_dataset)} samples from raddet_by_mr dataset")

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset
    
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=8, drop_last=True)
    
    logging.info('Training with %d cubes' % len(train_dataset))
    return train_loader

