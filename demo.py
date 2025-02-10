import os
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.ICFARNet import ICFARNet, autocast
import h5py
import pdb
import json
import matplotlib.pyplot as plt
import shutil
import pickle
from core.datasets import extract_local_maxima
import cv2
import matplotlib.cm as cm

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def __viz_cube(radar_cube: np.ndarray, cube_vis_path: str, name: str, center_radar_cube = None, rmtree = False, db = False, 
               gca=True, s=3, range_idx_list=None, azimuth_idx_list=None, class_name_list=None):
    
    '''
    可视化雷达立方体数据的函数
        radar_cube: (num_range_bins, num_velocity_bins, num_azimuth_bins) 256,512,12
    '''
    (num_range_bins, num_velocity_bins, num_azimuth_bins) = radar_cube.shape
    
    if not db:
        vis_path = os.path.join(cube_vis_path, name)
    else:
        vis_path = os.path.join(cube_vis_path, name+'_db') # _rcs
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)
    else:
        if rmtree:
            shutil.rmtree(vis_path)
            os.makedirs(vis_path)

    if db:
        img = 10 * np.log10(np.sum(radar_cube, axis=2)/radar_cube.shape[2] + 1e-10) - 20
    else:
        img = np.sum(radar_cube, axis=2)/radar_cube.shape[2]

    plt.imshow(img, cmap="jet")
    if gca:
        plt.gca().set_aspect(0.1)
    # 关闭坐标轴
    plt.axis('off')  # 关闭坐标轴
    # 保存图像
    plt.savefig(os.path.join(vis_path, "range-doppler_mean_.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    if db:
        img = 10 * np.log10(np.sum(radar_cube, axis=0)/radar_cube.shape[0] + 1e-10) - 20
    else:
        img = np.sum(radar_cube, axis=0)/radar_cube.shape[0]

    plt.imshow(img, cmap="jet")
    if gca:
        plt.gca().set_aspect(0.1)
    # 关闭坐标轴
    plt.axis('off')  # 关闭坐标轴
    # 保存图像
    plt.savefig(os.path.join(vis_path, "doppler-azimuth_mean_.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    if db:
        img = 10 * np.log10(np.max(radar_cube, axis=2) + 1e-10) - 20
    else:
        img = np.max(radar_cube, axis=2)

    plt.imshow(img, cmap="jet")
    if gca:
        plt.gca().set_aspect(0.1)
    # 关闭坐标轴
    plt.axis('off')  # 关闭坐标轴
    # 保存图像
    plt.savefig(os.path.join(vis_path, "range-doppler_max_.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    if db:
        img = 10 * np.log10(np.max(radar_cube, axis=0) + 1e-10) - 20
    else:
        img = np.max(radar_cube, axis=0)

    plt.imshow(img, cmap="jet")
    if gca:
        plt.gca().set_aspect(0.1)
    # 关闭坐标轴
    plt.axis('off')  # 关闭坐标轴
    # 保存图像
    plt.savefig(os.path.join(vis_path, "doppler-azimuth_max_.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    if db:
        img = 10 * np.log10(np.sum(radar_cube, axis=1)/radar_cube.shape[1] + 1e-10) - 20
    else:
        img = np.sum(radar_cube, axis=1)/radar_cube.shape[1]

    plt.imshow(img, cmap="jet")
    if gca:
        plt.gca().set_aspect(0.1)
    # 关闭坐标轴
    plt.axis('off')  # 关闭坐标轴
    # 保存图像
    plt.savefig(os.path.join(vis_path, "range-azimuth_mean_.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(img, cmap="jet")
    if gca:
        plt.gca().set_aspect(0.1)

    plt.colorbar()
    plt.xlabel("Azimuth")
    plt.ylabel("Range")
    plt.title("all Velocity mean")
    if range_idx_list is not None:
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'pink']
        markers = ['o', 's', '^', 'D', '*', 'x', '+', 'P', 'v', '<', '>']
        for cmap_idx, (range_idx, azimuth_idx, class_name) in enumerate(zip(range_idx_list, azimuth_idx_list, class_name_list)):
            plt.scatter(azimuth_idx, range_idx, c=colors[cmap_idx%(len(colors))], marker=markers[cmap_idx%(len(markers))], label='points')
             # 在每个散点的位置添加类名
            plt.text(azimuth_idx, range_idx, class_name, fontsize=9, ha='right', va='bottom', color='black')
    plt.savefig(os.path.join(vis_path, "range-azimuth_mean.png"), dpi=300)
    plt.close()

    if db:
        img = 10 * np.log10(np.max(radar_cube, axis=1) + 1e-10) - 20
    else:
        img = np.max(radar_cube, axis=1)

    plt.imshow(img, cmap="jet")
    if gca:
        plt.gca().set_aspect(0.1)
    # 关闭坐标轴
    plt.axis('off')  # 关闭坐标轴
    # 保存图像
    plt.savefig(os.path.join(vis_path, "range-azimuth_max_.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(img, cmap="jet")
    if gca:
        plt.gca().set_aspect(0.1)
    plt.colorbar()
    plt.xlabel("Azimuth")
    plt.ylabel("Range")
    plt.title("all Velocity max")
    if range_idx_list is not None:
        colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'pink']
        markers = ['o', 's', '^', 'D', '*', 'x', '+', 'P', 'v', '<', '>']
        for cmap_idx, (range_idx, azimuth_idx, class_name) in enumerate(zip(range_idx_list, azimuth_idx_list, class_name_list)):
            plt.scatter(azimuth_idx, range_idx, c=colors[cmap_idx%(len(colors))], marker=markers[cmap_idx%(len(markers))], label='points')
             # 在每个散点的位置添加类名
            plt.text(azimuth_idx, range_idx, class_name, fontsize=9, ha='right', va='bottom', color='black')
    plt.savefig(os.path.join(vis_path, "range-azimuth_max.png"), dpi=300)
    plt.close()

    if center_radar_cube is not None:
        for i in range(0, num_velocity_bins):
            if np.sum(center_radar_cube[:, i, :]) == 0:
                continue
            if db:
                img = 10 * np.log10(radar_cube[:, i, :] + 1e-10) - 20
            else:
                img = radar_cube[:, i, :]
            
            plt.imshow(img, cmap="jet")
            if gca:
                plt.gca().set_aspect(0.1)
            # 关闭坐标轴
            plt.axis('off')  # 关闭坐标轴
            # 保存图像
            plt.savefig(os.path.join(vis_path, f"range-azimuth_{i}_.png"), dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            plt.imshow(img, cmap="jet")
            if gca:
                plt.gca().set_aspect(0.1)
            plt.colorbar()
            plt.xlabel("Azimuth")
            plt.ylabel("Range")
            plt.title(f"Velocity idx: {i}")

            if range_idx_list is not None:
                colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'pink']
                markers = ['o', 's', '^', 'D', '*', 'x', '+', 'P', 'v', '<', '>']
                for cmap_idx, (range_idx, azimuth_idx) in enumerate(zip(range_idx_list, azimuth_idx_list)):
                    plt.scatter(azimuth_idx, range_idx, c=colors[cmap_idx%(len(colors))], marker=markers[cmap_idx%(len(markers))], label='points')

            plt.savefig(os.path.join(vis_path, f"range-azimuth_{i}.png"), dpi=300)
            plt.close()

def demo(args):
    model = ICFARNet(args) # torch.nn.DataParallel(ICFARNet(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model.cuda()
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory = output_directory / args.restore_ckpt.split('/')[-2]
    output_directory.mkdir(exist_ok=True)
    class_list = ["person", "bicycle", "car", "motorcycle", "bus", "truck" ]

    with torch.no_grad():
        RAD_path_list = sorted(glob.glob(f'{args.radar_path}/{args.version}/RAD/*/*.npy', recursive=True))

        print(f"Found {len(RAD_path_list)} images. Saving files to {output_directory}/")

        for RAD_path in tqdm(RAD_path_list):
            part_name = RAD_path.split('/')[-2]
            cube_name = RAD_path.split('/')[-1]

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
            radar_point_np = radar_point
            radar_point = radar_point * radar_cube_gt
            
            segment_mask = np.zeros_like(radar_cube_gt)
            gt_path = RAD_path.replace('/RAD/', '/gt/').replace('.npy', '.pickle')
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
                    segment_mask[x_begin:x_end, z_begin:z_end, y_begin:y_end] = class_list.index(class_now) + 1.0
            
            point_mask = radar_point > 0
            segment_mask = segment_mask * point_mask

            radar_point = torch.from_numpy(radar_point).float().unsqueeze(0).unsqueeze(0).cuda()
            segment_mask = torch.from_numpy(segment_mask).float().unsqueeze(0).unsqueeze(0).cuda()

            with autocast(enabled=args.mixed_precision):
                radar_cube = model(radar_point, segment_mask)

            radar_cube = radar_cube.squeeze(0).squeeze(0).cpu().numpy()

            range_idx_list=[] 
            azimuth_idx_list=[]
            class_name_list=[]
            real_sparse_radar_cube = np.zeros_like(radar_cube_gt)
            for idx, box in enumerate(radar_instances["boxes"]):
                range_idx_list.append(int(box[0]))
                azimuth_idx_list.append(int(box[1]))
                class_name_list.append(radar_instances["classes"][idx])
                real_sparse_radar_cube[int(box[0]),int(box[2]),int(box[1])] = 1.0
            
            RAD_name = RAD_path.split('/')[-2]+'-'+RAD_path.split('/')[-1].split('.')[0]

            if args.save_numpy:
                os.makedirs(os.path.join(output_directory, args.version), exist_ok=True)
                save_RAD_dir = os.path.join(output_directory, args.version, 'RAD')
                save_gt_dir = os.path.join(output_directory, args.version, 'gt')
                os.makedirs(save_RAD_dir, exist_ok=True)
                os.makedirs(save_gt_dir, exist_ok=True)
                os.makedirs(os.path.join(save_RAD_dir, part_name), exist_ok=True)
                os.makedirs(os.path.join(save_gt_dir, part_name), exist_ok=True)
                
                RAD_data = radar_cube.transpose((0, 2, 1))
                gt_instances = radar_instances
                if len(gt_instances["classes"]) != 0:
                    np.save(os.path.join(save_RAD_dir, part_name, cube_name), RAD_data)
                    with open(os.path.join(save_gt_dir, part_name, cube_name.replace('.npy', '.pickle')), 'wb') as f:
                        pickle.dump(gt_instances, f)
            else:
                __viz_cube(radar_cube_gt, output_directory, f"{RAD_name}_radar_cube_gt-real_box_center", real_sparse_radar_cube, rmtree = True, db = False, gca=False, s=1, range_idx_list=range_idx_list, azimuth_idx_list=azimuth_idx_list, class_name_list=class_name_list)
                __viz_cube(radar_cube, output_directory, f"{RAD_name}_radar_cube-real_box_center", real_sparse_radar_cube, rmtree = True, db = False, gca=False, s=1, range_idx_list=range_idx_list, azimuth_idx_list=azimuth_idx_list, class_name_list=class_name_list)
                print('radar_cube_icfar epe:', np.mean(np.abs(radar_cube-radar_cube_gt)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./models/icfar-net.pth')

    parser.add_argument('--radar_path', help="path to all radar_cube", default='/gpfs/essfs/iat/Tsinghua/xiaowq/RADDet')
    parser.add_argument("--version", type=str, default="train", help="train or test.")
    parser.add_argument('--output_directory', help="directory to save output", default="./sim_output/")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # ICFARNet Settings    
    parser.add_argument('--hidden_dims', type=int, default=32, help="hidden_dims.")
    parser.add_argument('--output_dims', type=int, default=1, help="output_dims.")
    
    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    demo(args)