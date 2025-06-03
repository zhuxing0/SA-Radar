import os
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.ICFARNet import ICFARNet, autocast
import matplotlib.pyplot as plt
import shutil
import pickle
from core.datasets import extract_local_maxima, compute_the_Attribute
import random

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def __viz_cube(radar_cube: np.ndarray, cube_vis_path: str, name: str, center_radar_cube = None, s=3, range_idx_list=None, azimuth_idx_list=None, class_name_list=None):
    
    (num_range_bins, num_velocity_bins, num_azimuth_bins) = radar_cube.shape
    class_name_to_icon = {"person":('blue', '^'), "bicycle":('cyan', 'D'), "car":('red', 'o'), "motorcycle":('magenta','*'), "bus":('yellow', 'x'), "truck":('green', 's')}

    vis_path = os.path.join(cube_vis_path, name)
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    img = np.sum(radar_cube, axis=2)/radar_cube.shape[2]
    plt.imshow(img, cmap="jet")
    plt.axis('off')
    plt.savefig(os.path.join(vis_path, "range-doppler_mean.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = np.sum(radar_cube, axis=0)/radar_cube.shape[0]
    plt.imshow(img, cmap="jet")
    plt.axis('off')
    plt.savefig(os.path.join(vis_path, "doppler-azimuth_mean.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = np.max(radar_cube, axis=2)
    plt.imshow(img, cmap="jet")
    plt.axis('off')
    plt.savefig(os.path.join(vis_path, "range-doppler_max.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


    img = np.max(radar_cube, axis=0)
    plt.imshow(img, cmap="jet")
    plt.axis('off')
    plt.savefig(os.path.join(vis_path, "doppler-azimuth_max.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()


    img = np.sum(radar_cube, axis=1)/radar_cube.shape[1]
    plt.imshow(img, cmap="jet")
    plt.axis('off')
    plt.savefig(os.path.join(vis_path, "range-azimuth_mean.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(img, cmap="jet")
    plt.colorbar()
    plt.xlabel("Azimuth")
    plt.ylabel("Range")
    plt.title("all Velocity mean")
    if range_idx_list is not None:
        for cmap_idx, (range_idx, azimuth_idx, class_name) in enumerate(zip(range_idx_list, azimuth_idx_list, class_name_list)):
            plt.scatter(azimuth_idx, range_idx, c=class_name_to_icon[class_name][0], marker=class_name_to_icon[class_name][1], label='points', s=20)
            text_x = azimuth_idx
            text_y = range_idx
            if text_x > img.shape[1] - 10:
                text_x = img.shape[1] - 10
            if text_y > img.shape[0] - 10:
                text_y = img.shape[0] - 10
            ha = 'left' if text_x < img.shape[1] - 10 else 'right'
            va = 'top' if text_y < img.shape[0] - 10 else 'bottom'
            plt.text(text_x, text_y, class_name, ha=ha, va=va, color='black', fontsize=15)
    plt.savefig(os.path.join(vis_path, "range-azimuth_mean_with_label.png"), dpi=300)
    plt.close()


    img = np.max(radar_cube, axis=1)
    plt.imshow(img, cmap="jet")
    plt.axis('off')
    plt.savefig(os.path.join(vis_path, "range-azimuth_max.png"), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    plt.imshow(img, cmap="jet")
    plt.colorbar()
    plt.xlabel("Azimuth")
    plt.ylabel("Range")
    plt.title("all Velocity max")
    if range_idx_list is not None:
        for cmap_idx, (range_idx, azimuth_idx, class_name) in enumerate(zip(range_idx_list, azimuth_idx_list, class_name_list)):
            plt.scatter(azimuth_idx, range_idx, c=class_name_to_icon[class_name][0], marker=class_name_to_icon[class_name][1], label='points', s=20)
            text_x = azimuth_idx
            text_y = range_idx
            if text_x > img.shape[1] - 10:
                text_x = img.shape[1] - 10
            if text_y > img.shape[0] - 10:
                text_y = img.shape[0] - 10
            ha = 'left' if text_x < img.shape[1] - 10 else 'right'
            va = 'top' if text_y < img.shape[0] - 10 else 'bottom'
            plt.text(text_x, text_y, class_name, ha=ha, va=va, color='black', fontsize=15)
    plt.savefig(os.path.join(vis_path, "range-azimuth_max.png"), dpi=300)
    plt.close()

    if center_radar_cube is not None:
        for i in range(0, num_velocity_bins):
            if np.sum(center_radar_cube[:, i, :]) == 0:
                continue

            img = radar_cube[:, i, :]
            plt.imshow(img, cmap="jet")
            plt.axis('off')
            plt.savefig(os.path.join(vis_path, f"range-azimuth_{i}_.png"), dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            plt.imshow(img, cmap="jet")
            plt.colorbar()
            plt.xlabel("Azimuth")
            plt.ylabel("Range")
            plt.title(f"Velocity idx: {i}")

            if range_idx_list is not None:
                for cmap_idx, (range_idx, azimuth_idx) in enumerate(zip(range_idx_list, azimuth_idx_list)):
                    plt.scatter(azimuth_idx, range_idx, c=class_name_to_icon[class_name][0], marker=class_name_to_icon[class_name][1], label='points', s=20)
                    text_x = azimuth_idx
                    text_y = range_idx
                    if text_x > img.shape[1] - 10:
                        text_x = img.shape[1] - 10
                    if text_y > img.shape[0] - 10:
                        text_y = img.shape[0] - 10
                    ha = 'left' if text_x < img.shape[1] - 10 else 'right'
                    va = 'top' if text_y < img.shape[0] - 10 else 'bottom'
                    plt.text(text_x, text_y, class_name, ha=ha, va=va, color='black', fontsize=15)
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

        attribute_list = []
        for sigma_r_bias_idx in [2, 3, 4]: # range(5):
            sigma_r_bias = (sigma_r_bias_idx-2)*0.2 # -0.2 ~ 0.2
            for N_list_bias in [-1, 0, 1]: # range(-2, 3): # -2~2
                for num_log_a_bias in [-1, 0, 1]: # range(-1, 2): # -1~1
                    attribute_list.append([sigma_r_bias, N_list_bias, num_log_a_bias])

        for idx, RAD_path in enumerate(tqdm(RAD_path_list)):
            part_name = RAD_path.split('/')[-2]

            sigma_r_bias, N_list_bias, num_log_a_bias = attribute_list[idx%len(attribute_list)]

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

            range_idx_box, doppler_idx_box, azimuth_idx_box, class_idx_box = [], [], [], []
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
                    x_begin, x_end = max(0, x_begin), min(num_range_bins-1, x_end)
                    y_begin, y_end = max(0, y_begin), min(num_azimuth_bins-1, y_end)
                    z_begin, z_end = max(0, z_begin), min(num_velocity_bins-1, z_end)
                    segment_mask[x_begin:x_end, z_begin:z_end, y_begin:y_end] = class_list.index(class_now) + 1.0
                    range_idx_box.extend([x_begin, x_end, x_begin, x_end, x_begin, x_end, x_begin, x_end])
                    azimuth_idx_box.extend([y_begin, y_end, y_end, y_begin, y_begin, y_end, y_end, y_begin])
                    doppler_idx_box.extend([z_begin, z_end, z_begin, z_end, z_end, z_begin, z_end, z_begin])
                    class_idx_box.extend([class_list.index(class_now) + 1.0, class_list.index(class_now) + 1.0, class_list.index(class_now) + 1.0, class_list.index(class_now) + 1.0,
                                        class_list.index(class_now) + 1.0, class_list.index(class_now) + 1.0, class_list.index(class_now) + 1.0, class_list.index(class_now) + 1.0])
            
            range_idx, doppler_idx, azimuth_idx = np.where(radar_point > 0.0)
            num_range_bins = 256
            range_bins = np.arange(0, num_range_bins).astype(np.float64)
            range_bin_vals = (256 - range_bins) * 0.1953125 # 50, 37.5, 25, 12.5, 0

            num_azimuth_bins = 256
            azimuth_bins = np.arange(0, num_azimuth_bins).astype(np.float64)
            azimuth_bin_vals = (azimuth_bins - 128) * 85.87 / 128 # 0.006135923 # -85.87, -42.93, 0, 42.93, 85.87

            num_velocity_bins = 64
            velocity_bins = np.arange(0, num_velocity_bins).astype(np.float64)
            velocity_bin_vals = (velocity_bins-32) * 0.41968030701528203 # -13, -6.5, 0, 6.5, 13

            range_list = range_bin_vals[range_idx]
            doppler_list = velocity_bin_vals[doppler_idx]
            azimuth_list = azimuth_bin_vals[azimuth_idx]
            
            azimuth_rad = np.radians(azimuth_list)
            x_all = range_list * np.cos(azimuth_rad)
            y_all = range_list * np.sin(azimuth_rad)
            z_all = doppler_list
            intensity_all = radar_point[range_idx, doppler_idx, azimuth_idx]

            range_idx_box, doppler_idx_box, azimuth_idx_box, class_idx_box = np.array(range_idx_box), np.array(doppler_idx_box), np.array(azimuth_idx_box), np.array(class_idx_box)
            range_list_box = range_bin_vals[range_idx_box]
            doppler_list_box = velocity_bin_vals[doppler_idx_box]
            azimuth_list_box = azimuth_bin_vals[azimuth_idx_box]
            azimuth_rad_box = np.radians(azimuth_list_box)
            x_all_box = range_list_box * np.cos(azimuth_rad_box)
            y_all_box = range_list_box * np.sin(azimuth_rad_box)
            z_all_box = doppler_list_box

            if args.eye_shift:
                r_shift = random.randint(-10, 10)
                a_shift = random.randint(-10, 10)
                radar_point_new = np.zeros_like(radar_point)
                r_b_in_new, r_e_in_new = max(0, r_shift), min(num_range_bins, num_range_bins+r_shift)
                r_b, r_e = max(0, -r_shift), min(num_range_bins, num_range_bins-r_shift)

                a_b_in_new, a_e_in_new = max(0, a_shift), min(num_azimuth_bins, num_azimuth_bins+a_shift)
                a_b, a_e = max(0, -a_shift), min(num_azimuth_bins, num_azimuth_bins-a_shift)

                radar_point_new[r_b_in_new:r_e_in_new, :, a_b_in_new:a_e_in_new] = radar_point[r_b:r_e, :, a_b:a_e]
                radar_point = radar_point_new

                radar_instances_shift = {"boxes":[], "classes":[], 'cart_boxes':[]}
                for idx_box, box in enumerate(radar_instances["boxes"]):
                    box[0] = box[0] + r_shift
                    box[1] = box[1] + a_shift
                    if 0<=box[0] and box[0]<num_range_bins and 0<=box[1] and box[1]<num_azimuth_bins:
                        radar_instances_shift["boxes"].append(box)
                        radar_instances_shift["classes"].append(radar_instances["classes"][idx_box])
                        radar_instances_shift["cart_boxes"].append(radar_instances["cart_boxes"][idx_box%len(radar_instances["cart_boxes"])])

            elif args.angle_rotation:
                ar_size = args.angle_rotation_size
                # centerx, centery, viewangle = random.randint(-ar_size, ar_size), random.randint(-ar_size, ar_size), 0
                
                centerx, centery, viewangle = 0, ar_size, 0
                centerx = max(centerx, -np.min(x_all_box))

                radar_point_new = np.zeros_like(radar_point)
                radar_point_new[:50] = np.clip(radar_point[:50], 0, 4.5*4.5)

                x_all, y_all, x_all_box, y_all_box = x_all + centerx, y_all + centery, x_all_box + centerx, y_all_box + centery
                xy = np.stack([x_all, y_all], axis=-1)
                xy_box = np.stack([x_all_box, y_all_box], axis=-1)

                ranges = np.linalg.norm(xy, axis=-1) 
                azimuths = np.degrees(np.arctan2(xy[:, 1], xy[:, 0])) 
                ranges_box = np.linalg.norm(xy_box, axis=-1) 
                azimuths_box = np.degrees(np.arctan2(xy_box[:, 1], xy_box[:, 0])) 

                mask = (
                    (x_all >= 0)
                    & (ranges <= np.max(range_bin_vals))
                    & (azimuths <= np.max(azimuth_bin_vals))
                    & (azimuths >= np.min(azimuth_bin_vals))
                )
                
                range_idx = np.digitize(ranges, (range_bin_vals[1:] + range_bin_vals[:-1]) / 2)
                azimuth_idx = np.digitize(azimuths, (azimuth_bin_vals[1:] + azimuth_bin_vals[:-1]) / 2)
                range_idx, azimuth_idx, doppler_idx, intensity_all = range_idx[mask], azimuth_idx[mask], doppler_idx[mask], intensity_all[mask]
                for r_idx, a_idx, d_idx, intensity in zip(range_idx, azimuth_idx, doppler_idx, intensity_all):
                    radar_point_new[r_idx, d_idx, a_idx] = intensity
                radar_point = radar_point_new

                range_idx_box = np.digitize(ranges_box, (range_bin_vals[1:] + range_bin_vals[:-1]) / 2)
                azimuth_idx_box = np.digitize(azimuths_box, (azimuth_bin_vals[1:] + azimuth_bin_vals[:-1]) / 2)
                range_idx_box, doppler_idx_box, azimuth_idx_box, class_idx_box = range_idx_box.reshape(-1, 8), doppler_idx_box.reshape(-1, 8), azimuth_idx_box.reshape(-1, 8), class_idx_box.reshape(-1, 8)
                
                radar_instances_rotation = {"boxes":[], "classes":[], 'cart_boxes':[]}
                for idx_box, (r_idx, a_idx, d_idx, class_idx) in enumerate(zip(range_idx_box, azimuth_idx_box, doppler_idx_box, class_idx_box)):
                    r_idx, a_idx = sorted(r_idx[:4]), sorted(a_idx[:4])

                    x_begin, x_end = (r_idx[0]+r_idx[1])/2, (r_idx[2]+r_idx[3])/2
                    y_begin, y_end = (a_idx[0]+a_idx[1])/2, (a_idx[2]+a_idx[3])/2
                    z_begin, z_end = np.min(d_idx), np.max(d_idx)

                    box = [(x_begin+x_end)/2, (y_begin+y_end)/2, (z_begin+z_end)/2, x_end-x_begin, y_end-y_begin, z_end-z_begin]

                    class_idx = int(class_idx[0]-1)
                    radar_instances_rotation["boxes"].append(box)
                    radar_instances_rotation["classes"].append(class_list[class_idx])
                    radar_instances_rotation["cart_boxes"].append(radar_instances["cart_boxes"][idx_box%len(radar_instances["cart_boxes"])])
            
            elif args.remove:
                class_now = radar_instances["classes"][-1]
                xyzwhd = radar_instances["boxes"][-1]
                radar_instances["classes"] = radar_instances["classes"][:-1]
                radar_instances["boxes"] = radar_instances["boxes"][:-1]
                x_begin = int(xyzwhd[0] - xyzwhd[3]//2)
                x_end = int(xyzwhd[0] + xyzwhd[3]//2)
                y_begin = int(xyzwhd[1] - xyzwhd[4]//2)
                y_end = int(xyzwhd[1] + xyzwhd[4]//2)
                z_begin = int(xyzwhd[2] - xyzwhd[5]//2)
                z_end = int(xyzwhd[2] + xyzwhd[5]//2)
                x_begin, x_end = max(0, x_begin), min(num_range_bins, x_end)
                y_begin, y_end = max(0, y_begin), min(num_azimuth_bins, y_end)
                z_begin, z_end = max(0, z_begin), min(num_velocity_bins, z_end)
                radar_point[x_begin:x_end, z_begin:z_end, y_begin:y_end] = np.clip(radar_point[x_begin:x_end, z_begin:z_end, y_begin:y_end], 0, 4.5*4.5)


            radar_point = torch.from_numpy(radar_point).float().unsqueeze(0).unsqueeze(0).cuda()
            with autocast(enabled=args.mixed_precision):
                if args.attribute:
                    sigma, g, Rs, lambda_ = compute_the_Attribute(sigma_r_bias, N_list_bias, num_log_a_bias)
                    sigma_item, g_item, Rs_item, lambda_item = round(sigma,2), round(g,2), round(Rs,2), round(lambda_,2)
                    radar_cube_zero = torch.zeros_like(radar_point)
                    sigma, g, Rs, lambda_ = radar_cube_zero+sigma, radar_cube_zero+g, radar_cube_zero+Rs, radar_cube_zero+lambda_
                    radar_cube = model(radar_point, sigma, g, Rs, lambda_)
                else:
                    radar_cube = model(radar_point)
            radar_cube = radar_cube.squeeze(0).squeeze(0).cpu().numpy()

            RAD_name = RAD_path.split('/')[-2] + '-' + RAD_path.split('/')[-1].split('.')[0] + f'-sigma{sigma_item}_g{g_item}_Rs{Rs_item}_lambda{lambda_item}'

            if args.save_numpy:
                os.makedirs(os.path.join(output_directory, args.version), exist_ok=True)
                save_RAD_dir = os.path.join(output_directory, args.version, 'RAD')
                save_gt_dir = os.path.join(output_directory, args.version, 'gt')
                os.makedirs(save_RAD_dir, exist_ok=True)
                os.makedirs(save_gt_dir, exist_ok=True)
                os.makedirs(os.path.join(save_RAD_dir, part_name), exist_ok=True)
                os.makedirs(os.path.join(save_gt_dir, part_name), exist_ok=True)
                
                if args.eye_shift:
                    gt_instances = radar_instances_shift
                    RAD_name = RAD_name + f'_rshift{r_shift}_ashift{a_shift}'
                elif args.angle_rotation:
                    gt_instances = radar_instances_rotation
                    RAD_name = RAD_name + f'_centerx{centerx}_centery{centery}_angle{viewangle}'
                elif args.remove:
                    gt_instances = radar_instances
                    RAD_name = RAD_name + f'_remove'

                RAD_name = RAD_name + '.npy'
                if len(gt_instances["classes"]) != 0:
                    RAD_data = radar_cube.transpose((0, 2, 1))
                    np.save(os.path.join(save_RAD_dir, part_name, RAD_name), RAD_data)
                    with open(os.path.join(save_gt_dir, part_name, RAD_name.replace('.npy', '.pickle')), 'wb') as f:
                        pickle.dump(gt_instances, f)
            else:
                range_idx_list=[] 
                azimuth_idx_list=[]
                class_name_list=[]
                real_sparse_radar_cube = np.zeros_like(radar_cube_gt)
                for idx, box in enumerate(radar_instances["boxes"]):
                    range_idx_list.append(int(box[0]))
                    azimuth_idx_list.append(int(box[1]))
                    class_name_list.append(radar_instances["classes"][idx])
                    real_sparse_radar_cube[int(box[0]),int(box[2]),int(box[1])] = 1.0

                if args.eye_shift:
                    gt_instances = radar_instances_shift
                    RAD_name = RAD_name + f'_rshift{r_shift}_ashift{a_shift}'
                elif args.angle_rotation:
                    gt_instances = radar_instances_rotation
                    RAD_name = RAD_name + f'_centerx{centerx}_centery{centery}_angle{viewangle}'
                elif args.remove:
                    gt_instances = radar_instances
                    RAD_name = RAD_name + f'_remove'

                __viz_cube(radar_cube_gt, output_directory, f"{RAD_name}_radar_cube_gt-real_box_center", real_sparse_radar_cube, s=1, range_idx_list=range_idx_list, azimuth_idx_list=azimuth_idx_list, class_name_list=class_name_list)
                __viz_cube(radar_cube, output_directory, f"{RAD_name}_radar_cube-real_box_center", real_sparse_radar_cube, s=1, range_idx_list=range_idx_list, azimuth_idx_list=azimuth_idx_list, class_name_list=class_name_list)
                print('radar_cube_icfar epe:', np.mean(np.abs(radar_cube-radar_cube_gt)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./models/icfar-net.pth')

    parser.add_argument('--radar_path', help="path to all radar_cube", default='./dataset/RADDet')
    parser.add_argument("--version", type=str, default="train", help="train or test.")
    parser.add_argument('--output_directory', help="directory to save output", default="./sim_output/")
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')

    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--eye_shift', action='store_true', help='Viewpoint RA shift') 
    parser.add_argument('--angle_rotation', action='store_true', help='Novel Trajectories')
    parser.add_argument('--angle_rotation_size', type=int, default=5, help="Novel Trajectories Size.")
    parser.add_argument('--remove', action='store_true', help='Actor Removal') 

    # ICFARNet Settings    
    parser.add_argument('--hidden_dims', type=int, default=32, help="hidden_dims.")
    parser.add_argument('--output_dims', type=int, default=1, help="output_dims.")
    parser.add_argument('--attribute', action='store_true', help="attribute embedding or not")

    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    demo(args)