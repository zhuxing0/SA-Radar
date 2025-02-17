import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pdb
import shutil
import os
import time
import matplotlib.pyplot as plt
import pickle
import cv2
import glob
from scipy.ndimage import maximum_filter

def generate_radar_cube_by_camera(xy_list: list, doppler_list: list, rcs_list: list, range_bin_vals, doppler_bin_vals, azimuth_bin_vals):
    '''
        This function receives the 2D position (xy), velocity (doppler), and radar cross-section (rcs) of every target, 
        and generates a pseudo radar cube.
        
        inputs:
            xy_list: [xy1, xy2, ..., xyn], where xyn's shape is (N, 3) or (N, 2).
            doppler_list: [doppler1, doppler2, ..., dopplern], where dopplern's shape is (N, 1).
            rcs_list: [rcs1, rcs2, ..., rcsn], where rcsn's shape is (N, 1).
            range_bin_vals: (num_range_bins,), representing the actual values of radar_cube in the range dimension.
            doppler_bin_vals: (num_doppler_bins,), representing the actual values of radar_cube in the doppler dimension.
            azimuth_bin_vals: (num_azimuth_bins,), representing the actual values of radar_cube in the azimuth dimension.
        output:
            pseudo_radar_cube: (num_range_bins, num_doppler_bins, num_azimuth_bins)
    '''

    num_range_bins, num_doppler_bins, num_azimuth_bins = range_bin_vals.shape[0], doppler_bin_vals.shape[0], azimuth_bin_vals.shape[0]
    range_bins = np.arange(0, num_range_bins).astype(np.float64)
    doppler_bins = np.arange(0, num_doppler_bins).astype(np.float64)

    pseudo_radar_cube = np.zeros((num_range_bins, num_doppler_bins, num_azimuth_bins), dtype=np.float32)

    # Gaussian parameters
    sigma_r = 2.6  # Standard deviation of the range Gaussian
    A_r = 1.0  # Amplitude of the range Gaussian

    # Generate some global noise points, can be changed to a pre-generated mode (i.e., "pre-saved locally, read randomly") for acceleration
    for doppler_i in range(0, num_doppler_bins):
        num_points = np.random.randint(530, 550)  # Generate random points

        noise_range_idx = (np.arange(num_points)*num_range_bins/num_points).astype(int)
        noise_azimuth_idx = (np.random.uniform(low=0, high=num_azimuth_bins, size=num_points)).astype(int)
        noise_rcs = np.random.normal(3.752+0.15, 0.3565-0.15, size=num_points) 
        N_list = np.random.randint(8, 10, size=num_points)
        num_log_a_list = np.random.randint(2, 4, size=num_points)
        a_bios_list = np.random.uniform(1.0, 1.5, size=num_points)

        for idx, (range_i, azimuth_i, rcs) in tqdm(enumerate(zip(noise_range_idx, noise_azimuth_idx, noise_rcs))):

            noise_rcs_max = rcs

            # Calculate the Gaussian response in the range direction
            PSF_r = A_r * np.exp(-((range_bins - range_i) ** 2) / (2 * sigma_r ** 2))
            
            PSF_d = 0.0
            adjust_v_list = [(-1, 0.3), (0, 1), (0, 1), (0, 1), (1, 0.3)]
            for adjust_v, intensity in adjust_v_list:
                signal_v = np.exp(1j * 2 * np.pi * ((doppler_i+adjust_v)/num_doppler_bins) * doppler_bins)
                PSF_d += intensity * np.abs(np.fft.fft(signal_v))/100

            # The length of the window can be adjusted as needed; the longer the window, the narrower the main lobe and the side lobes
            N = max(1, int(N_list[idx] * num_azimuth_bins / 256)) 
            pangban = 0.1
            window = np.arange(0, N).astype(np.float64)
            window = (1-pangban)-pangban*np.cos(2*np.pi*window/N-1)
            # Calculate the Fourier transform of the window
            spectrum = np.fft.fft(window, n=num_azimuth_bins)  # Use zero-padding to 1024 points
            spectrum = np.fft.fftshift(spectrum)  # Center the spectrum
            PSF_a = np.abs(np.roll(spectrum, azimuth_i-num_azimuth_bins//2))
            '''
                Control the peak ratio between the main lobe and side lobes using log10 and a_bios, specific experimental results are as follows:
                num_log_a-a_bios: 2-1.0=0.716; 3-1.0=0.876; 3-1.0=0.876; 3-1.5=0.914; 4-1.0=0.951
                Reference actual values: 0.88/0.80/0.74/0.87/0.90, etc.
            '''
            for _ in range(num_log_a_list[idx]):
                PSF_a = 10 * np.log10(PSF_a + a_bios_list[idx])

            # Calculate the attenuation factor ratio_i related to the range, used to simulate the attenuation of signal strength. 
            # The attenuation factor is inversely proportional to the fourth power of distance, which is common in radar signal processing.
            # However, we found that in real radar datasets (like RADDet), the radar cube does not show significant intensity attenuation (suspected to be removed in preprocessing), so we set ratio_i = 1.0
            ratio_i = 1.0  # ratio_i = 1.0 / (range_i**4) 

            # This step is the main reason for the slow code.
            PSF_r, PSF_d, PSF_a = PSF_r-np.min(PSF_r), PSF_d-np.min(PSF_d), PSF_a-np.min(PSF_a)
            pseudo_radar_cube_noise_point = ratio_i * PSF_r[:, None, None] * PSF_d[None, :, None] * PSF_a[None, None, :]
            pseudo_radar_cube_noise_point = noise_rcs_max / np.max(pseudo_radar_cube_noise_point) * pseudo_radar_cube_noise_point

            mask = 5 * (pseudo_radar_cube_noise_point - pseudo_radar_cube)
            mask = 1 / (1 + np.exp(-mask))
            pseudo_radar_cube = (1-mask)*pseudo_radar_cube + mask*pseudo_radar_cube_noise_point  

    # --------------------------------------end normal noise setting process--------------------------------------

    # -----------------------------Reflection point convolution----------------------------- #
    for xy, doppler, rcs in zip(xy_list, doppler_list, rcs_list):
        # x is front, y is left
        ranges = np.linalg.norm(xy, axis=-1)  # Calculate the distance of each target to the origin.
        azimuths = np.arctan2(xy[:, 1], xy[:, 0])  # Calculate the azimuth angle of each target.
        velocities = doppler.reshape(-1)

        # Map the range values to the corresponding indices. 
        range_idx = np.digitize(ranges, (range_bin_vals[1:] + range_bin_vals[:-1]) / 2)
        # Map the doppler values to the corresponding indices.
        doppler_idx = np.digitize(
            velocities, (doppler_bin_vals[1:] + doppler_bin_vals[:-1]) / 2
        )
        # Map the azimuth values to the corresponding indices.
        azimuth_idx = np.digitize(
            azimuths, (azimuth_bin_vals[1:] + azimuth_bin_vals[:-1]) / 2
        )
        pseudo_radar_cube_box = np.zeros_like(pseudo_radar_cube)

        num_points = len(range_idx)
        N_list = np.random.randint(8, 10, size=num_points)
        num_log_a_list = np.random.randint(2, 4, size=num_points)
        a_bios_list = np.random.uniform(1.0, 1.5, size=num_points)

        for idx, (range_i, doppler_i, azimuth_i, rcs_max) in enumerate(zip(range_idx, doppler_idx, azimuth_idx, rcs)):

            # Calculate the Gaussian response in the range direction
            PSF_r = A_r * np.exp(-((range_bins - range_i) ** 2) / (2 * sigma_r ** 2))
            
            PSF_d = 0.0
            adjust_v_list = [(-1, 0.3), (0, 1), (0, 1), (0, 1), (1, 0.3)]
            for adjust_v, intensity in adjust_v_list:
                signal_v = np.exp(1j * 2 * np.pi * ((doppler_i+adjust_v)/num_doppler_bins) * doppler_bins)
                PSF_d += intensity * np.abs(np.fft.fft(signal_v))/100

            # The length of the window can be adjusted as needed; the longer the window, the narrower the main lobe and the side lobes
            N = max(1, int(N_list[idx] * num_azimuth_bins / 256)) 
            pangban = 0.1
            window = np.arange(0, N).astype(np.float64)
            window = (1-pangban)-pangban*np.cos(2*np.pi*window/N-1)
            # Calculate the Fourier transform of the window
            spectrum = np.fft.fft(window, n=num_azimuth_bins)  # Use zero-padding to 1024 points
            spectrum = np.fft.fftshift(spectrum)  # Center the spectrum
            PSF_a = np.abs(np.roll(spectrum, azimuth_i-num_azimuth_bins//2))
            '''
                Control the peak ratio between the main lobe and side lobes using log10 and a_bios, specific experimental results are as follows:
                num_log_a-a_bios: 2-1.0=0.716; 3-1.0=0.876; 3-1.0=0.876; 3-1.5=0.914; 4-1.0=0.951
                Reference actual values: 0.88/0.80/0.74/0.87/0.90, etc.
            '''
            for _ in range(num_log_a_list[idx]):
                PSF_a = 10 * np.log10(PSF_a + a_bios_list[idx])

            # Calculate the attenuation factor ratio_i related to the range, used to simulate the attenuation of signal strength. 
            # The attenuation factor is inversely proportional to the fourth power of distance, which is common in radar signal processing.
            # However, we found that in real radar datasets (like RADDet), the radar cube does not show significant intensity attenuation (suspected to be removed in preprocessing), so we set ratio_i = 1.0
            ratio_i = 1.0  # ratio_i = 1.0 / (range_i**4) 

            PSF_r, PSF_d, PSF_a = PSF_r-np.min(PSF_r), PSF_d-np.min(PSF_d), PSF_a-np.min(PSF_a)
            pseudo_radar_cube_box += ratio_i * PSF_r[:, None, None] * PSF_d[None, :, None] * PSF_a[None, None, :]
            pseudo_radar_cube_box = rcs_max / np.max(pseudo_radar_cube_box) * pseudo_radar_cube_box

            mask = 5 * (pseudo_radar_cube_box - pseudo_radar_cube)
            mask = 1 / (1 + np.exp(-mask))
            pseudo_radar_cube = (1-mask)*pseudo_radar_cube + mask*pseudo_radar_cube_box
    # -----------------------------end reflection point convolution----------------------------- #

    return pseudo_radar_cube

def generate_radar_cube_by_real_cube(real_cube):

    num_range_bins, num_doppler_bins, num_azimuth_bins = real_cube.shape
    pseudo_radar_cube = np.zeros((num_range_bins, num_doppler_bins, num_azimuth_bins), dtype=np.float32)

    range_bins = np.arange(0, num_range_bins).astype(np.float64)
    doppler_bins = np.arange(0, num_doppler_bins).astype(np.float64)

    # 高斯参数
    sigma_r = 2.6  # 距离高斯的标准差
    A_r = 1.0  # 距离高斯的幅度

    for doppler_i in range(num_doppler_bins):
        ra_slice = real_cube[:, doppler_i, :]
        img_ref = extract_local_maxima(ra_slice)
        noise_range_idx, noise_azimuth_idx = np.where(img_ref > 0.0)
        mask_in = (noise_azimuth_idx != 0) & (noise_azimuth_idx != 255)
        noise_range_idx, noise_azimuth_idx = noise_range_idx[mask_in], noise_azimuth_idx[mask_in]

        num_points = len(noise_range_idx)
        N_list = np.random.randint(8, 10, size=num_points)
        num_log_a_list = np.random.randint(2, 4, size=num_points)
        a_bios_list = np.random.uniform(1.0, 1.5, size=num_points)
        
        for idx, (range_i, azimuth_i) in enumerate(zip(noise_range_idx, noise_azimuth_idx)):

            noise_rcs_max = img_ref[range_i, azimuth_i]

            # 计算距离方向的高斯响应
            PSF_r = A_r * np.exp(-((range_bins - range_i) ** 2) / (2 * sigma_r ** 2))
            
            PSF_d = 0.0
            adjust_v_list = [(-1, 0.3), (0, 1), (0, 1), (0, 1), (1, 0.3)]
            for adjust_v, intensity in adjust_v_list:
                signal_v = np.exp(1j * 2 * np.pi * ((doppler_i+adjust_v)/num_doppler_bins) * doppler_bins)
                PSF_d += intensity * np.abs(np.fft.fft(signal_v))/100

            # 可以根据需要调整窗的长度, 窗越长，主瓣越窄, 旁瓣也越窄
            N = max(1, int(N_list[idx] * num_azimuth_bins / 256)) 
            pangban = 0.1
            window = np.arange(0, N).astype(np.float64)
            window = (1-pangban)-pangban*np.cos(2*np.pi*window/N-1)
            # 计算窗的傅里叶变换
            spectrum = np.fft.fft(window, n=num_azimuth_bins)  # 使用零填充到1024点
            spectrum = np.fft.fftshift(spectrum)  # 将频谱中心化
            PSF_a = np.abs(np.roll(spectrum, azimuth_i-num_azimuth_bins//2))
            '''
                通过 log10 和 a_bios 来控制主瓣和旁瓣之间的峰值比, 具体实验结果如下：
                num_log_a-a_bios: 2-1.0=0.716; 3-1.0=0.876; 3-1.0=0.876; 3-1.5=0.914; 4-1.0=0.951
                参考实际数值: 0.88/0.80/0.74/0.87/0.90等
            '''
            for _ in range(num_log_a_list[idx]):
                PSF_a = 10 * np.log10(PSF_a + a_bios_list[idx])

            # Calculate the attenuation factor ratio_i related to the range, used to simulate the attenuation of signal strength. 
            # The attenuation factor is inversely proportional to the fourth power of distance, which is common in radar signal processing.
            # However, we found that in real radar datasets (like RADDet), the radar cube does not show significant intensity attenuation (suspected to be removed in preprocessing), so we set ratio_i = 1.0
            ratio_i = 1.0  # ratio_i = 1.0 / (range_i**4) 

            # 这一步是代码慢的主要原因。
            PSF_r, PSF_d, PSF_a = PSF_r-np.min(PSF_r), PSF_d-np.min(PSF_d), PSF_a-np.min(PSF_a)
            pseudo_radar_cube_noise_point = ratio_i * PSF_r[:, None, None] * PSF_d[None, :, None] * PSF_a[None, None, :]
            pseudo_radar_cube_noise_point = noise_rcs_max / np.max(pseudo_radar_cube_noise_point) * pseudo_radar_cube_noise_point

            mask = 5 * (pseudo_radar_cube_noise_point - pseudo_radar_cube)
            mask = 1 / (1 + np.exp(-mask))
            pseudo_radar_cube = (1-mask)*pseudo_radar_cube + mask*pseudo_radar_cube_noise_point  

    return pseudo_radar_cube

def generate_radar_cube_by_real_cube_fastv1(real_cube, real_bboxs, bg_npy_dir):

    num_range_bins, num_doppler_bins, num_azimuth_bins = real_cube.shape
    pseudo_radar_cube = np.zeros((num_range_bins, num_doppler_bins, num_azimuth_bins), dtype=np.float32)

    range_bins = np.arange(0, num_range_bins).astype(np.float64)
    doppler_bins = np.arange(0, num_doppler_bins).astype(np.float64)

    # 高斯参数
    sigma_r = 2.6  # 距离高斯的标准差
    A_r = 1.0  # 距离高斯的幅度

    bg_npy_list = os.listdir(bg_npy_dir)

    for doppler_i in range(num_doppler_bins):
        bg_idx = np.random.randint(0, len(bg_npy_list))
        bg_npy_path = os.path.join(bg_npy_dir, bg_npy_list[bg_idx].replace('.jpg', '.npy'))
        bg_npy = np.load(bg_npy_path)
        
        pseudo_radar_cube_noise_point = np.zeros((num_range_bins, num_doppler_bins+4, num_azimuth_bins), dtype=np.float32)
        pseudo_radar_cube_noise_point[:, doppler_i-2+2:doppler_i+3+2, :] = bg_npy
        pseudo_radar_cube_noise_point = pseudo_radar_cube_noise_point[:, 2:num_doppler_bins+2, :]

        mask = 5 * (pseudo_radar_cube_noise_point - pseudo_radar_cube)
        mask = 1 / (1 + np.exp(-mask))
        pseudo_radar_cube = (1-mask)*pseudo_radar_cube + mask*pseudo_radar_cube_noise_point  

    bboxs_radar_cube = np.zeros((num_range_bins, num_doppler_bins, num_azimuth_bins), dtype=np.float32)
    for xyzwhd in real_bboxs:
        x_begin = int(xyzwhd[0] - xyzwhd[3]//2)
        x_end = int(xyzwhd[0] + xyzwhd[3]//2)
        y_begin = int(xyzwhd[1] - xyzwhd[4]//2)
        y_end = int(xyzwhd[1] + xyzwhd[4]//2)
        z_begin = int(xyzwhd[2] - xyzwhd[5]//2)
        z_end = int(xyzwhd[2] + xyzwhd[5]//2)
        x_begin, x_end = max(0, x_begin), min(num_range_bins, x_end)
        y_begin, y_end = max(0, y_begin), min(num_azimuth_bins, y_end)
        z_begin, z_end = max(0, z_begin), min(num_doppler_bins, z_end)
        bboxs_radar_cube[x_begin:x_end, z_begin:z_end, y_begin:y_end] = 1.0

    for doppler_i in range(num_doppler_bins):
        ra_slice = real_cube[:, doppler_i, :]
        img_ref = extract_local_maxima(ra_slice)
        noise_range_idx, noise_azimuth_idx = np.where(img_ref*bboxs_radar_cube[:, doppler_i, :] > 0.0)
        mask_in = (noise_azimuth_idx != 0) & (noise_azimuth_idx != 255)
        noise_range_idx, noise_azimuth_idx = noise_range_idx[mask_in], noise_azimuth_idx[mask_in]

        num_points = len(noise_range_idx)
        N_list = np.random.randint(8, 10, size=num_points)
        num_log_a_list = np.random.randint(2, 4, size=num_points)
        a_bios_list = np.random.uniform(1.0, 1.5, size=num_points)
        
        for idx, (range_i, azimuth_i) in enumerate(zip(noise_range_idx, noise_azimuth_idx)):

            noise_rcs_max = img_ref[range_i, azimuth_i]

            # 计算距离方向的高斯响应
            PSF_r = A_r * np.exp(-((range_bins - range_i) ** 2) / (2 * sigma_r ** 2))
            
            PSF_d = 0.0
            adjust_v_list = [(-1, 0.3), (0, 1), (0, 1), (0, 1), (1, 0.3)]
            for adjust_v, intensity in adjust_v_list:
                signal_v = np.exp(1j * 2 * np.pi * ((doppler_i+adjust_v)/num_doppler_bins) * doppler_bins)
                PSF_d += intensity * np.abs(np.fft.fft(signal_v))/100

            # 可以根据需要调整窗的长度, 窗越长，主瓣越窄, 旁瓣也越窄
            N = max(1, int(N_list[idx] * num_azimuth_bins / 256)) 
            pangban = 0.1
            window = np.arange(0, N).astype(np.float64)
            window = (1-pangban)-pangban*np.cos(2*np.pi*window/N-1)
            # 计算窗的傅里叶变换
            spectrum = np.fft.fft(window, n=num_azimuth_bins)  # 使用零填充到1024点
            spectrum = np.fft.fftshift(spectrum)  # 将频谱中心化
            PSF_a = np.abs(np.roll(spectrum, azimuth_i-num_azimuth_bins//2))
            '''
                通过 log10 和 a_bios 来控制主瓣和旁瓣之间的峰值比, 具体实验结果如下：
                num_log_a-a_bios: 2-1.0=0.716; 3-1.0=0.876; 3-1.0=0.876; 3-1.5=0.914; 4-1.0=0.951
                参考实际数值: 0.88/0.80/0.74/0.87/0.90等
            '''
            for _ in range(num_log_a_list[idx]):
                PSF_a = 10 * np.log10(PSF_a + a_bios_list[idx])

            # Calculate the attenuation factor ratio_i related to the range, used to simulate the attenuation of signal strength. 
            # The attenuation factor is inversely proportional to the fourth power of distance, which is common in radar signal processing.
            # However, we found that in real radar datasets (like RADDet), the radar cube does not show significant intensity attenuation (suspected to be removed in preprocessing), so we set ratio_i = 1.0
            ratio_i = 1.0  # ratio_i = 1.0 / (range_i**4) 

            # 这一步是代码慢的主要原因。
            PSF_r, PSF_d, PSF_a = PSF_r-np.min(PSF_r), PSF_d-np.min(PSF_d), PSF_a-np.min(PSF_a)
            pseudo_radar_cube_noise_point = ratio_i * PSF_r[:, None, None] * PSF_d[None, :, None] * PSF_a[None, None, :]
            pseudo_radar_cube_noise_point = noise_rcs_max / np.max(pseudo_radar_cube_noise_point) * pseudo_radar_cube_noise_point

            mask = 5 * (pseudo_radar_cube_noise_point - pseudo_radar_cube)
            mask = 1 / (1 + np.exp(-mask))
            pseudo_radar_cube = (1-mask)*pseudo_radar_cube + mask*pseudo_radar_cube_noise_point  

    return pseudo_radar_cube

def extract_local_maxima(img):
    output = np.zeros_like(img)
    neighborhood = maximum_filter(img, size=3)
    local_max = (img == neighborhood)
    output[local_max] = img[local_max]
    return output

def main(args):

    version = str(args.version)
    radar_path = Path(args.radar_path)
    save_dir = Path(os.path.join(args.output_directory, args.save_dir)) 
    save_dir.mkdir(parents=True, exist_ok=True)

    save_split_dir = save_dir / version
    save_split_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(__file__, save_split_dir)

    gt_save_path = save_split_dir / 'gt'
    RAD_save_path = save_split_dir / 'RAD'
    os.makedirs(gt_save_path, exist_ok=True)
    os.makedirs(RAD_save_path, exist_ok=True)

    npy_list = sorted(glob.glob(f"{radar_path}/{version}/RAD/*/*.npy"))

    for real_cube_path in tqdm(npy_list):
        time_start_generate = time.time()

        part_name = real_cube_path.split('/')[-2]
        cube_name = real_cube_path.split('/')[-1].split('.')[0]
        real_label_path = f"{radar_path}/{version}/gt/{part_name}/{cube_name}.pickle"

        if os.path.exists(real_label_path):
            with open(real_label_path, "rb") as f:
                radar_instances = pickle.load(f)
            if len(radar_instances['classes']) == 0:
                radar_instances = None
        else:
            radar_instances = None
        # print(cube_name, 'real_label:', radar_instances)

        real_cube = np.load(real_cube_path)
        real_cube = np.abs(real_cube)
        real_cube = pow(real_cube, 2)
        real_cube = np.log10(real_cube + 1.)
        real_cube = real_cube.transpose((0, 2, 1))

        # pseudo_radar_cube = generate_radar_cube_by_real_cube(real_cube)
        pseudo_radar_cube = generate_radar_cube_by_real_cube_fastv1(real_cube, radar_instances["boxes"], 'Pre-generated_data/bg_npy_v2')

        RAD_data = pseudo_radar_cube.transpose((0, 2, 1))

        gt_instances = radar_instances
        if len(gt_instances["classes"]) != 0:
            np.save(RAD_save_path / f"{part_name}-{cube_name}.npy", RAD_data) # 
            with open(gt_save_path / f"{part_name}-{cube_name}.pickle", 'wb') as f:
                pickle.dump(gt_instances, f)

        # print('sim_label:', gt_instances)

        time_end_generate = time.time()
        print(f"  generate radar cube '{part_name}-{cube_name}' time: {time_end_generate - time_start_generate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar_path", type=str, default="/gpfs/essfs/iat/Tsinghua/xiaowq/RADDet", help="RADDet dataset path.")
    parser.add_argument("--version", type=str, default="train", help="train or test.")
    parser.add_argument('--output_directory', help="directory to save output", default="./sim_output/")
    parser.add_argument("--save_dir", type=str, default="EXsim_onRADDAT", help="sim save filename")
    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    main(args)
