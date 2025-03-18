import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
import shutil
import os
import time
import pickle
import glob
from scipy.ndimage import maximum_filter

def generate_radar_cube_by_real_cube_fastv1(real_cube, real_bboxs, bg_npy_dir, bg_npy_list, sigma_r_bias, N_list_bias, num_log_a_bias, g_bias):

    num_range_bins, num_doppler_bins, num_azimuth_bins = real_cube.shape
    pseudo_radar_cube = np.zeros((num_range_bins, num_doppler_bins, num_azimuth_bins), dtype=np.float32)

    range_bins = np.arange(0, num_range_bins).astype(np.float64)
    doppler_bins = np.arange(0, num_doppler_bins).astype(np.float64)

    radar_dict = {'sigma_r': 2.6+sigma_r_bias,

                  'N_list_min': 8+N_list_bias,
                  'N_list_max': 10+N_list_bias, # 窗越长，主瓣越窄, 旁瓣也越窄
                  'num_log_a_list_min': 2+num_log_a_bias, # num_log_a-a_bias都与旁瓣能量占比成正相关
                  'num_log_a_list_max': 4+num_log_a_bias,
                  'a_bias_list_min': 1,
                  'a_bias_list_max': 1.5,
                  'pangban': 0.1
                  }
    
    sigma_r = radar_dict['sigma_r']

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

        N_list = np.random.randint(radar_dict['N_list_min'], radar_dict['N_list_max'], size=num_points)
        num_log_a_list = np.random.randint(radar_dict['num_log_a_list_min'], radar_dict['num_log_a_list_max'], size=num_points)
        a_bias_list = np.random.uniform(radar_dict['a_bias_list_min'], radar_dict['a_bias_list_max'], size=num_points)
        
        for idx, (range_i, azimuth_i) in enumerate(zip(noise_range_idx, noise_azimuth_idx)):

            noise_rcs_max = img_ref[range_i, azimuth_i]

            # 计算距离方向的高斯响应
            PSF_r = np.exp(-((range_bins - range_i) ** 2) / (2 * sigma_r ** 2))
            
            PSF_d = 0.0
            adjust_v_list = [(-1, 0.3), (0, 1), (0, 1), (0, 1), (1, 0.3)]
            for adjust_v, intensity in adjust_v_list:
                signal_v = np.exp(1j * 2 * np.pi * ((doppler_i+adjust_v)/num_doppler_bins) * doppler_bins)
                PSF_d += intensity * np.abs(np.fft.fft(signal_v))/100

            PSF_d = np.clip(PSF_d * g_bias - (g_bias-1)*max(PSF_d), 0, 10000)

            # 可以根据需要调整窗的长度, 窗越长，主瓣越窄, 旁瓣也越窄
            N = max(1, int(N_list[idx] * num_azimuth_bins / 256)) 
            pangban = radar_dict['pangban']
            window = np.arange(0, N).astype(np.float64)
            window = (1-pangban)-pangban*np.cos(2*np.pi*window/N-1)
            # 计算窗的傅里叶变换
            spectrum = np.fft.fft(window, n=num_azimuth_bins)  # 使用零填充到1024点
            spectrum = np.fft.fftshift(spectrum)  # 将频谱中心化
            PSF_a = np.abs(np.roll(spectrum, azimuth_i-num_azimuth_bins//2))
            '''
                通过 log10 和 a_bias 来控制主瓣和旁瓣之间的峰值比, 具体实验结果如下：
                num_log_a-a_bias: 2-1.0=0.716; 3-1.0=0.876; 3-1.0=0.876; 3-1.5=0.914; 4-1.0=0.951
                参考实际数值: 0.88/0.80/0.74/0.87/0.90等
            '''
            for _ in range(num_log_a_list[idx]):
                PSF_a = 10 * np.log10(PSF_a + a_bias_list[idx])

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

    bg_npy_dir = 'Pre-generated_data/bg_npy_mixed'
    bg_npy_list_all = os.listdir(bg_npy_dir)
    radar_list = []
    for g_bias in [0.8, 1, 1.2]:
        for sigma_r_bias_idx in range(5):
            sigma_r_bias = (sigma_r_bias_idx-2)*0.1 # -0.2 ~ 0.2
            for N_list_bias in range(-2, 3): # -2~2
                for num_log_a_bias in range(-1, 2): # -1~1
                    bg_npy_list = []
                    for bg_npy_name in bg_npy_list_all:
                        if '.npy' in bg_npy_name:
                            sigma_r_bias_now, N_list_bias_now, num_log_a_bias_now, g_bias_now = float(bg_npy_name.split('_')[0]), float(bg_npy_name.split('_')[1]), float(bg_npy_name.split('_')[2]), float(bg_npy_name.split('_')[3])
                            if np.abs(sigma_r_bias_now-sigma_r_bias)<0.05 and np.abs(N_list_bias_now-N_list_bias)<0.05 and np.abs(num_log_a_bias_now-num_log_a_bias)<0.05 and np.abs(g_bias_now-g_bias)<0.05:
                                bg_npy_list.append(bg_npy_name)
                    if len(bg_npy_list) == 0:
                        continue
                    radar_list.append((bg_npy_list, sigma_r_bias, N_list_bias, num_log_a_bias, g_bias))

    for RAD_index, real_cube_path in enumerate(tqdm(npy_list)):
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

        bg_npy_list, sigma_r_bias, N_list_bias, num_log_a_bias, g_bias = radar_list[RAD_index%len(radar_list)]

        pseudo_radar_cube = generate_radar_cube_by_real_cube_fastv1(real_cube, radar_instances["boxes"], bg_npy_dir, bg_npy_list, sigma_r_bias, N_list_bias, num_log_a_bias, g_bias)
        
        RAD_data = pseudo_radar_cube.transpose((0, 2, 1))

        gt_instances = radar_instances
        if len(gt_instances["classes"]) != 0:
            np.save(RAD_save_path / f"{part_name}_{cube_name}_{sigma_r_bias}_{N_list_bias}_{num_log_a_bias}_{g_bias}.npy", RAD_data) # 
            with open(gt_save_path / f"{part_name}_{cube_name}_{sigma_r_bias}_{N_list_bias}_{num_log_a_bias}_{g_bias}.pickle", 'wb') as f:
                pickle.dump(gt_instances, f)

        time_end_generate = time.time()
        print(f"  generate radar cube '{part_name}-{cube_name}' time: {time_end_generate - time_start_generate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--radar_path", type=str, default="dataset/RADDet", help="RADDet dataset path.")
    parser.add_argument("--version", type=str, default="train", help="train or test.")
    parser.add_argument('--output_directory', help="directory to save output", default="./sim_output/")
    parser.add_argument("--save_dir", type=str, default="EXsim_onRADDAT_multi-radar", help="sim save filename")
    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    main(args)
