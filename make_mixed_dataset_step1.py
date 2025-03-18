import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def generate_radar_cube_of_bg(bg_npy_path, g_bias, sigma_r_bias, N_list_bias, num_log_a_bias, rcs_bias, bg_npy_num=20):

    num_range_bins, num_doppler_bins, num_azimuth_bins = 256, 64, 256
    pseudo_radar_cube = np.zeros((num_range_bins, num_doppler_bins, num_azimuth_bins), dtype=np.float32)

    range_bins = np.arange(0, num_range_bins).astype(np.float64)
    doppler_bins = np.arange(0, num_doppler_bins).astype(np.float64)

    radar_dict = {'num_points_min': 530,
                  'num_points_max': 550,

                  'noise_rcs_mean': 3.90+rcs_bias,
                  'noise_rcs_sd': 0.2065,

                  'sigma_r': 2.6+sigma_r_bias,

                  'N_list_min': 8+N_list_bias,
                  'N_list_max': 10+N_list_bias, # 窗越长，主瓣越窄, 旁瓣也越窄
                  'num_log_a_list_min': 2+num_log_a_bias, # num_log_a-a_bias都与旁瓣能量占比成正相关
                  'num_log_a_list_max': 4+num_log_a_bias,
                  'a_bias_list_min': 1,
                  'a_bias_list_max': 1.5,
                  'pangban': 0.1
                  }

    # 高斯参数
    sigma_r = radar_dict['sigma_r']  # 距离高斯的标准差
    
    bg_npy_num_now = 0
    bg_npy_index_to_cube = str(sigma_r_bias)+'_'+str(N_list_bias)+'_'+str(num_log_a_bias)+'_'+str(g_bias)

    exist_bg_list = os.listdir(bg_npy_path)
    if f"{bg_npy_index_to_cube}_{bg_npy_num-1}.jpg" in exist_bg_list:
        return
    
    print(radar_dict)

    for doppler_i in range(0, num_doppler_bins): # range(23, 26): # [31, 32, 33]: # range(0, num_doppler_bins):
            
        # --------------------------------------正常设置噪声过程--------------------------------------

        num_points = np.random.randint(radar_dict['num_points_min'], radar_dict['num_points_max']) # np.random.randint(560, 580)

        # 生成随机点
        noise_range_idx = (np.arange(num_points)*num_range_bins/num_points).astype(int)
        noise_azimuth_idx = (np.random.uniform(low=0, high=num_azimuth_bins, size=num_points)).astype(int)
        noise_rcs = np.random.normal(radar_dict['noise_rcs_mean'], radar_dict['noise_rcs_sd'], size=num_points) # np.random.uniform(4.0, 4.5, size=num_points)
        N_list = np.random.randint(radar_dict['N_list_min'], radar_dict['N_list_max'], size=num_points)
        num_log_a_list = np.random.randint(radar_dict['num_log_a_list_min'], radar_dict['num_log_a_list_max'], size=num_points)
        a_bias_list = np.random.uniform(radar_dict['a_bias_list_min'], radar_dict['a_bias_list_max'], size=num_points)

        pseudo_radar_cube_noise_point_patch = np.zeros_like(pseudo_radar_cube)
        for idx, (range_i, azimuth_i, rcs) in tqdm(enumerate(zip(noise_range_idx, noise_azimuth_idx, noise_rcs))):

            noise_rcs_max = rcs

            # 计算距离方向的高斯响应
            PSF_r = np.exp(-((range_bins - range_i) ** 2) / (2 * sigma_r ** 2))
            
            PSF_d = 0.0
            adjust_v_list = [(-1, 0.3), (0, 3), (1, 0.3)]
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

            ratio_i = 1.0

            PSF_r, PSF_d, PSF_a = PSF_r-np.min(PSF_r), PSF_d-np.min(PSF_d), PSF_a-np.min(PSF_a)
            pseudo_radar_cube_noise_point = ratio_i * PSF_r[:, None, None] * PSF_d[None, :, None] * PSF_a[None, None, :]
            pseudo_radar_cube_noise_point = noise_rcs_max / np.max(pseudo_radar_cube_noise_point) * pseudo_radar_cube_noise_point

            mask = 5 * (pseudo_radar_cube_noise_point - pseudo_radar_cube_noise_point_patch)
            mask = 1 / (1 + np.exp(-mask))
            pseudo_radar_cube_noise_point_patch = (1-mask)*pseudo_radar_cube_noise_point_patch + mask*pseudo_radar_cube_noise_point

        if doppler_i >= 2 and doppler_i < num_doppler_bins-3:
            pseudo_radar_cube_noise_point_patch = pseudo_radar_cube_noise_point_patch[:,doppler_i-2:doppler_i+3,:]

            np.save(os.path.join(bg_npy_path, f"{bg_npy_index_to_cube}_{bg_npy_num_now}.npy"), pseudo_radar_cube_noise_point_patch)
        
            img = pseudo_radar_cube_noise_point_patch[:,2,:]
            plt.imshow(img, cmap="jet", vmin=0)
            plt.colorbar()
            plt.xlabel("Azimuth")
            plt.ylabel("Range")
            plt.savefig(os.path.join(bg_npy_path, f"{bg_npy_index_to_cube}_{bg_npy_num_now}.jpg"), dpi=300)
            plt.close()

            bg_npy_num_now += 1
            if bg_npy_num_now >= bg_npy_num:
                return
            
if __name__ == '__main__':

    bg_npy_path = 'Pre-generated_data/bg_npy_mixed'
    os.makedirs(bg_npy_path, exist_ok=True)

    radar_list = []
    for g_bias in [0.8, 1, 1.2]:
        for sigma_r_bias_idx in range(5):
            sigma_r_bias = (sigma_r_bias_idx-2)*0.1 # -0.2 ~ 0.2
            for N_list_bias in range(-2, 3): # -2~2
                for num_log_a_bias in range(-1, 2): # -1~1
                        rcs_bias = np.random.uniform(-1, 1)
                        radar_list.append((g_bias, sigma_r_bias, N_list_bias, num_log_a_bias, rcs_bias))
    
    for (g_bias, sigma_r_bias, N_list_bias, num_log_a_bias, rcs_bias) in tqdm(radar_list):
        generate_radar_cube_of_bg(bg_npy_path, g_bias, sigma_r_bias, N_list_bias, num_log_a_bias, rcs_bias, bg_npy_num=10)