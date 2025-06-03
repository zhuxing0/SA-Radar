import argparse
import json
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes, RadarPointCloud, LidarPointCloud
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from tqdm import tqdm
import pdb
import shutil
import os
import time
import matplotlib.pyplot as plt
import pickle
from core.ICFARNet import ICFARNet, autocast
import torch
from core.datasets import extract_local_maxima, compute_the_Attribute

def __viz_cube(radar_cube: np.ndarray, cube_vis_path: str, name: str, center_radar_cube = None, s=3, range_idx_list=None, azimuth_idx_list=None, class_name_list=None):
    
    (num_range_bins, num_doppler_bins, num_azimuth_bins) = radar_cube.shape
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
        for i in range(0, num_doppler_bins):
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

def calculate_box_corners(translation, size, rotation):
    """
    Calculate the 8 corner points of the box and apply a 90-degree rotation.
    """
    l, w, h = size
    center = translation
    rotation_quat = Quaternion(rotation)
    additional_rotation = Quaternion(axis=[0, 0, 1], angle=np.pi / 2)  # 90 degrees in radians
    combined_rotation = rotation_quat * additional_rotation
    box = np.array([[-l / 2, -w / 2, -h / 2],
                        [l / 2, -w / 2, -h / 2],
                        [l / 2, w / 2, -h / 2],
                        [-l / 2, w / 2, -h / 2],
                        [-l / 2, -w / 2, h / 2],
                        [l / 2, -w / 2, h / 2],
                        [l / 2, w / 2, h / 2],
                        [-l / 2, w / 2, h / 2]])
    
    box = (combined_rotation.rotation_matrix @ box.T).T + center[:3]
    return box

def is_points_in_box(points, box):
    """
    Check if the given points are inside the box.

    Parameters:
    - points: A NumPy array of shape (N, 3) representing the x, y, z coordinates of N points.
    - box: A NumPy array of shape (8, 3) representing the x, y, z coordinates of the 8 corner points of the box.
    # Example data
    box = np.array([[-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1]])
    Returns:
    - mask: A boolean array of shape (N,) indicating which points are inside the box.
    """
    bottom_points = box[:4]
    top_points = box[4:]

    v1 = bottom_points[1] - bottom_points[0]
    v2 = bottom_points[3] - bottom_points[0]
    normal = np.cross(v1, v2)

    normal = normal / np.linalg.norm(normal)

    point_vectors = points[:, np.newaxis, :] - bottom_points[0]  # (N, 1, 3)
    distances = np.dot(point_vectors, normal)  # (N, 1)

    below_bottom = distances < 0

    point_vectors_top = points[:, np.newaxis, :] - top_points[0]  # (N, 1, 3)
    distances_top = np.dot(point_vectors_top, -normal)  # (N, 1)

    above_top = distances_top < 0

    inside_bottom = np.zeros((points.shape[0], 4), dtype=bool)
    for i in range(4):
        v1 = bottom_points[(i + 1) % 4] - bottom_points[i]
        v2 = points - bottom_points[i] 
        cross_products = np.cross(v1, v2)
        inside_bottom[:, i] = np.dot(cross_products, normal) >= 0 

    mask = ~below_bottom.ravel() & ~above_top.ravel() & np.all(inside_bottom, axis=1)
    return mask


def main(args):
    np.random.seed(0)

    model = ICFARNet(args)

    checkpoint = torch.load(args.restore_ckpt)
    model.load_state_dict(checkpoint, strict=True)
    
    model.cuda()
    model.eval()

    time_start = time.time()

    version = str(args.version)
    data_dir = Path(args.data_dir)

    attribute_list = []
    for sigma_r_bias_idx in [2, 3, 4]:
        sigma_r_bias = (sigma_r_bias_idx-2)*0.2
        for N_list_bias in [-1, 0, 1]:
            for num_log_a_bias in [-1, 0, 1]:
                attribute_list.append([sigma_r_bias, N_list_bias, num_log_a_bias])

    if not args.attribute:
        attribute_list = [[0,0,0]]

    for attribute_idx, attributes in enumerate(attribute_list):

        (sigma_r_bias, N_list_bias, num_log_a_bias) = attributes

        if args.attribute:
            save_dir = Path(args.save_dir) / (version + f'_attribute_idx-{attribute_idx}')
        else:
            save_dir = Path(args.save_dir) / version

        save_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(__file__, save_dir)

        nusc = NuScenes(version=version, dataroot=data_dir, verbose=False)
        lidarseg_idx2name_mapping = nusc.lidarseg_idx2name_mapping

        time_get_nusc = time.time()
        print(f"Time to get nusc: {time_get_nusc - time_start:.2f} seconds")

        num_range_bins = 256
        range_bins = np.arange(0, num_range_bins).astype(np.float64)
        range_bin_vals = (256 - range_bins) * 0.1953125 # 50, 37.5, 25, 12.5, 0

        num_azimuth_bins = 256
        azimuth_bins = np.arange(0, num_azimuth_bins).astype(np.float64)
        azimuth_bin_vals = (azimuth_bins - 128) * 85.87 / 128 # 0.006135923 # -85.87, -42.93, 0, 42.93, 85.87
        azimuth_bin_vals = np.radians(azimuth_bin_vals)

        num_doppler_bins = 64
        doppler_bins = np.arange(0, num_doppler_bins).astype(np.float64)
        doppler_bin_vals = (doppler_bins-32) * 0.41968030701528203 # -13, -6.5, 0, 6.5, 13

        save_path_raddet = save_dir / Path('simRADDET-time_steps_'+str(args.time_steps))
        save_path_raddet.mkdir(parents=True, exist_ok=True)
        gt_save_path = save_path_raddet / 'gt'
        if not os.path.exists(gt_save_path):
            os.makedirs(gt_save_path)
        RAD_save_path = save_path_raddet / 'RAD'
        if not os.path.exists(RAD_save_path):
            os.makedirs(RAD_save_path)

        class_list = ["person", "bicycle", "car", "motorcycle", "bus", "truck" ]
        
        for scene in tqdm(nusc.scene):
            print(scene["description"])
            sample_token_curr = scene["first_sample_token"]
            scene_data = {}
            '''
                scene_data = {'channel1': {'timestamp11':{}, 'timestamp12':{}, ...}, 'channel2': {'timestamp21':{}, 'timestamp22':{}, ...}, ...}
            '''
            cnt = 0

            save_path = save_dir / Path(str(scene["name"]) +'-'+str(args.time_steps)) / 'RADAR_ALL'
            save_path.mkdir(parents=True, exist_ok=True)

            while True:
                cnt += 1
                if args.time_steps == 1 and cnt > 2:
                    break
                elif args.time_steps != 1 and args.time_steps != 0 and cnt > args.time_steps:
                    break
                
                sample = nusc.get("sample", sample_token_curr)

                # Collect all LiDAR points, and group them by 3D annotation box
                for channel in sample["data"].keys():
                    sample_data_token = sample["data"][channel]
                    sample_data = nusc.get("sample_data", sample_data_token)
                    if "lidar" in channel.lower():
                        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
                        sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
                        T_ego2global = transform_matrix(ego_pose["translation"],Quaternion(ego_pose["rotation"]),inverse=False,)
                        T_sensor2ego = transform_matrix(sensor["translation"], Quaternion(sensor["rotation"]), inverse=False)
                        T_sensor2global = T_ego2global @ T_sensor2ego

                        lidar_pcl_path = os.path.join(data_dir, sample_data['filename'])
                        lidar_pc = LidarPointCloud.from_file(lidar_pcl_path)

                        lidar_xy = lidar_pc.points[:3].T

                        ones = np.ones((lidar_xy.shape[0], 1))
                        lidar_xy_homogeneous = np.hstack((lidar_xy, ones))
                        lidar_xy_global = (T_sensor2global @ lidar_xy_homogeneous.T).T

                        z_height = 1.0
                        z_tolerance = 1.5
                        mask_height = np.abs(lidar_xy_global[:, 2] - z_height) < z_tolerance
                        lidar_xy_global = lidar_xy_global[mask_height]

                        boxes = []
                        for ann_token in sample['anns']:
                            annotation = nusc.get('sample_annotation', ann_token)
                            corners = calculate_box_corners(annotation['translation'], annotation['size'], annotation['rotation'])
                            boxes.append((annotation['category_name'], corners))

                        organized_points_lidar = {}
                        for (category_name, corners) in boxes:
                            mask_box = is_points_in_box(lidar_xy_global[:, :3], corners)
                            points_in_box = lidar_xy_global[mask_box]
                            if points_in_box.shape[0] == 0:
                                continue
                            if category_name not in organized_points_lidar:
                                organized_points_lidar[category_name] = []
                            points_in_box = np.array(points_in_box)
                            organized_points_lidar[category_name].append(points_in_box)
                    else:
                        continue

                # Translate Lidar Points to Radar Coordinate System
                for channel in sample["data"].keys():
                    sample_data_token = sample["data"][channel]
                    sample_data = nusc.get("sample_data", sample_data_token)
                    if "radar" in channel.lower():
                        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
                        sensor = nusc.get(
                            "calibrated_sensor", sample_data["calibrated_sensor_token"]
                        )
                        T_ego2global = transform_matrix(
                            ego_pose["translation"],
                            Quaternion(ego_pose["rotation"]),
                            inverse=False,
                        )
                        T_sensor2ego = transform_matrix(
                            sensor["translation"], Quaternion(sensor["rotation"]), inverse=False
                        )

                        T_sensor2global = T_ego2global @ T_sensor2ego
                        T_sensor2global[2, 3] = 1.0
                        T_global2sensor = np.linalg.inv(T_sensor2global)

                        organized_points_now = {}
                        for category_name in organized_points_lidar.keys():
                            points_in_box_list = organized_points_lidar[category_name]
                            for points_in_box in points_in_box_list:
                                xy = (T_global2sensor @ points_in_box.T).T
                                mask = (xy.T)[0] > 0
                                xy = xy[mask]

                                if len(xy) > 0:
                                    if category_name not in organized_points_now:
                                        organized_points_now[category_name] = []
                                    organized_points_now[category_name].append(xy[:, :3])

                        timestamp = sample_data["timestamp"] / 1e6
                        if channel not in scene_data:
                            scene_data[channel] = {}
                        scene_data[channel][timestamp] = {
                            "organized_points_now": organized_points_now,
                            "T_sensor2global": T_sensor2global,
                        }
                    else:
                        continue

                sample_token_curr = sample["next"]

                if sample_token_curr == "":
                    break
            
            name2idx_mapping = {}
            for k, v in lidarseg_idx2name_mapping.items():
                name2idx_mapping[v] = k

            # reflection intensity mapping
            mapping_dict = {
                0: (None, 0.4, 0.5, 'normal'),
                1: (None, 0.4, 0.8, 'normal'),
                2: ("person", 0.4, 0.8, 'normal'),
                3: ("person", 0.4, 0.8, 'normal'),
                4: ("person", 0.4, 0.8, 'normal'),
                5: ("person", 0.4, 0.8, 'normal'),
                6: ("person", 0.4, 0.8, 'normal'),
                7: ("person", 0.4, 0.8, 'normal'),
                8: ("person", 0.4, 0.8, 'normal'),
                9: (None, 0.4, 0.6, 'normal'),
                10: (None, 0.4, 0.6, 'normal'),
                11: (None, 0.4, 0.6, 'normal'),
                12: (None, 0.60, 0.85, 'normal'),
                13: ("bicycle", 0.45, 0.90, 'normal'),
                14: ("bicycle", 0.45, 0.90, 'normal'),
                15: ("bus", 0.60, 1.00, 'normal'),
                16: ("bus", 0.60, 1.00, 'normal'),
                17: ("car", 0.40, 1.00, 'normal'),
                18: ("truck", 0.40, 1.00, 'normal'),
                19: ("bus", 0.60, 1.00, 'normal'),
                20: ("car", 0.40, 1.00, 'normal'),
                21: ("motorcycle", 0.60, 0.85, 'uniform'),
                22: ("truck", 0.40, 1.00, 'normal'),
                23: ("truck", 0.40, 1.00, 'normal'),
                24: (None, 0.4, 0.6, 'normal'),
                25: (None, 0.4, 0.6, 'normal'),
                26: (None, 0.4, 0.6, 'normal'),
                27: (None, 0.4, 0.6, 'normal'),
                28: (None, 0.50, 0.70, 'normal'),
                29: (None, 0.50, 0.70, 'normal'),
                30: (None, 0.4, 0.6, 'normal'),
                31: (None, 0.4, 0.6, 'normal'),
            }

            for channel, channel_data in scene_data.items():
                channel_data = dict(sorted(channel_data.items(), key=lambda x: x[0]))
                timestamps = list(channel_data.keys())

                # Handles only the front radar (optional)
                if channel != 'RADAR_FRONT':
                    continue

                print(' Processing radar cube of channel {}'.format(channel))

                for i, t in enumerate(timestamps):
                    np.random.seed(0)
                    if args.time_steps == 1 and i != 0:
                        continue
                    
                    time_start_generate = time.time()

                    T_sensor2global = channel_data[t]["T_sensor2global"]

                    x = T_sensor2global[:3, 3]
                    A = T_sensor2global[:3, :3]

                    # Calculate the sensor's velocity in the global coordinate system based on the timestamps.
                    if i < len(timestamps) - 1:
                        vel = (
                            channel_data[timestamps[i + 1]]["T_sensor2global"][:3, 3] - x
                        ) / (timestamps[i + 1] - t)
                    else:
                        vel = (
                            x - channel_data[timestamps[i - 1]]["T_sensor2global"][:3, 3]
                        ) / (t - timestamps[i - 1])

                    v = np.linalg.inv(A) @ vel # Converts the sensor velocity vector from the global coordinate system to the sensor coordinate system.
                    s = np.linalg.norm(v)
                    v = np.nan_to_num(v / s, nan=0.0, posinf=0.0, neginf=0.0)
                    p = np.array([1, 0, 0]) - v[0] * v 
                    p /= np.linalg.norm(p)
                    q = np.cross(v, p)
                    q /= np.linalg.norm(q)

                    vel_sensor = (A.T @ vel)[:3].reshape(1, -1)

                    xy_list, intensity_list, doppler_list, segment_list = [], [], [], []
                    gt_instances = {"classes":[], "boxes":[]}
                    organized_points_now = channel_data[t]["organized_points_now"]
                    for category_name in organized_points_now.keys():
                        points_in_box_list = organized_points_now[category_name]
                        for points_in_box in points_in_box_list:
                            if len(points_in_box) < 5:
                                continue

                            doppler_in_box = -1 * (np.sum(vel_sensor * points_in_box, axis=-1) / np.linalg.norm(points_in_box, axis=-1)).reshape(-1, 1)
                            ranges = np.linalg.norm(points_in_box, axis=-1)
                            azimuths = np.arctan2(points_in_box[:, 1], points_in_box[:, 0])
                            doppler_in_box = doppler_in_box.reshape(-1)

                            mask = (
                                (ranges <= np.max(range_bin_vals))
                                & (doppler_in_box <= np.max(doppler_bin_vals))
                                & (doppler_in_box >= np.min(doppler_bin_vals))
                                & (azimuths <= np.max(azimuth_bin_vals))
                                & (azimuths >= np.min(azimuth_bin_vals))
                            )

                            ranges, doppler_in_box, azimuths, points_in_box = (
                                ranges[mask],
                                doppler_in_box[mask],
                                azimuths[mask],
                                points_in_box[mask]
                            )

                            if ranges.shape[0] == 0:
                                continue

                            range_idx = np.digitize(ranges, (range_bin_vals[1:] + range_bin_vals[:-1]) / 2)
                            azimuth_idx = np.digitize(
                                azimuths, (azimuth_bin_vals[1:] + azimuth_bin_vals[:-1]) / 2
                            )

                            classes, intensity_min, intensity_max, dis_type = mapping_dict[name2idx_mapping[category_name]]

                            if classes is not None:
                                doppler_in_box = doppler_in_box - np.min(doppler_in_box)
                                doppler_in_box = doppler_in_box + np.random.uniform(doppler_bin_vals[1], doppler_bin_vals[-2]-np.max(doppler_in_box))

                                doppler_in_box = doppler_in_box + np.random.normal(0, 1, doppler_in_box.shape)*(doppler_bin_vals[1]-doppler_bin_vals[0])
                                doppler_in_box = np.clip(doppler_in_box, doppler_bin_vals[1], doppler_bin_vals[-2])
                            else:
                                doppler_in_box = doppler_in_box * 0.0

                            if classes is not None:
                                doppler_idx = np.digitize(
                                    doppler_in_box, (doppler_bin_vals[1:] + doppler_bin_vals[:-1]) / 2
                                )

                                x_center = (np.max(range_idx) + np.min(range_idx))/2
                                y_center = (np.max(azimuth_idx) + np.min(azimuth_idx))/2
                                z_center = (np.max(doppler_idx) + np.min(doppler_idx))/2
                                w = max(np.max(range_idx) - np.min(range_idx) + 2, 4)
                                h = max(np.max(azimuth_idx) - np.min(azimuth_idx) + 8, 4)
                                d = max(np.max(doppler_idx) - np.min(doppler_idx) + 1, 2)

                                gt_instances["boxes"].append([x_center, y_center, z_center, w, h, d])
                                gt_instances["classes"].append(classes)

                            if dis_type == 'normal':
                                intensity_in_box = np.random.normal(1, 0.05, size=len(doppler_in_box))*np.random.normal((intensity_min+intensity_max)/2, (intensity_max-intensity_min)/4.6)
                            elif dis_type == 'uniform':
                                intensity_in_box = np.random.normal(1, 0.05, size=len(doppler_in_box))*np.random.uniform(intensity_min, intensity_max)

                            intensity_in_box = np.clip(intensity_in_box, 0, 1)
                            xy_list.append(points_in_box)
                            doppler_list.append(doppler_in_box)
                            intensity_list.append(intensity_in_box)
                            if classes is not None:
                                segment_list.append(np.ones_like(intensity_in_box)*(class_list.index(classes) + 1.0))
                            else:
                                segment_list.append(np.zeros_like(intensity_in_box))
                            

                    if len(xy_list) == 0 or len(gt_instances["classes"]) == 0:
                        continue

                    sparse_radar_cube = np.zeros((num_range_bins, num_doppler_bins, num_azimuth_bins), dtype=np.float32)

                    # ----------- Generate random noise reflection points ----------------------#
                    for doppler_i in range(0, num_doppler_bins):
                        num_points = np.random.randint(530, 550)
                        noise_range_idx = (np.arange(num_points)*num_range_bins/num_points).astype(int)
                        noise_azimuth_idx = (np.random.uniform(low=0, high=num_azimuth_bins, size=num_points)).astype(int)
                        noise_rcs = np.random.normal(3.90, 0.2065, size=num_points)
                        for idx, (range_i, azimuth_i, rcs_i) in enumerate(zip(noise_range_idx, noise_azimuth_idx, noise_rcs)):
                            sparse_radar_cube[range_i, doppler_i, azimuth_i] = rcs_i
                    
                    # # ---------- Extracting noise points from real cube -------- #
                    # real_cube = np.load('xxxxx.npy')
                    # real_cube = pow(real_cube, 2)
                    # real_cube = np.log10(real_cube + 1.)
                    # radar_cube_gt = real_cube.transpose((0, 2, 1)) # 256, 64, 256
                    # for doppler_i in range(num_doppler_bins):
                    #     sparse_radar_cube[:,doppler_i,:] = extract_local_maxima(radar_cube_gt[:,doppler_i,:])

                    # --------------- Generate scene reflection points -------- #
                    for xy, velocity, rcs in zip(xy_list, doppler_list, intensity_list):
                        ranges = np.linalg.norm(xy, axis=-1)
                        azimuths = np.arctan2(xy[:, 1], xy[:, 0])
                        velocities = velocity.reshape(-1)

                        range_idx = np.digitize(ranges, (range_bin_vals[1:] + range_bin_vals[:-1]) / 2)
                        doppler_idx = np.digitize(velocities, (doppler_bin_vals[1:] + doppler_bin_vals[:-1]) / 2)
                        azimuth_idx = np.digitize(azimuths, (azimuth_bin_vals[1:] + azimuth_bin_vals[:-1]) / 2)

                        sparse_radar_cube[range_idx, doppler_idx, azimuth_idx] = rcs*10
                    
                    radar_point = sparse_radar_cube * sparse_radar_cube
                    radar_point = torch.from_numpy(radar_point).float().unsqueeze(0).unsqueeze(0).cuda()
                    with autocast(enabled=args.mixed_precision):
                        if args.attribute:
                            sigma, g, Rs, lambda_ = compute_the_Attribute(sigma_r_bias, N_list_bias, num_log_a_bias)
                            radar_cube_zero = torch.zeros_like(radar_point)
                            sigma, g, Rs, lambda_ = radar_cube_zero+sigma, radar_cube_zero+g, radar_cube_zero+Rs, radar_cube_zero+lambda_
                            radar_cube = model(radar_point, sigma, g, Rs, lambda_)
                        else:
                            radar_cube = model(radar_point)

                    radar_cube = radar_cube.squeeze(0).squeeze(0).cpu().detach().numpy()

                    RAD_data = radar_cube.transpose((0, 2, 1))
                    gt_instances["boxes"] = np.array(gt_instances["boxes"])
                    if len(gt_instances["classes"]) != 0:
                        RAD_name = str(scene["name"]) + f'_{channel}_frame-{i}'
                        np.save(RAD_save_path / f"{RAD_name}.npy", RAD_data)
                        with open(gt_save_path / f"{RAD_name}.pickle", 'wb') as f:
                            pickle.dump(gt_instances, f)

                    time_end_generate = time.time()
                    print(f"  generate_radar_cube of {i} time: {time_end_generate - time_start_generate}")

                    if args.vis and channel == 'RADAR_FRONT':
                        time_start_vis = time.time()
                        range_idx_list=[] 
                        azimuth_idx_list=[]
                        doppler_idx_list=[]
                        class_name_list=[]
                        real_sparse_radar_cube = np.zeros_like(radar_cube)
                        for idx, box in enumerate(gt_instances["boxes"]):
                            range_idx_list.append(int(box[0]))
                            azimuth_idx_list.append(int(box[1]))
                            doppler_idx_list.append(int(box[2]))
                            class_name_list.append(gt_instances["classes"][idx])
                            real_sparse_radar_cube[int(box[0]),int(box[2]),int(box[1])] = 1.0

                        __viz_cube(radar_cube, save_path, f"{RAD_name}_sim_radar_cube", real_sparse_radar_cube, s=1, range_idx_list=range_idx_list, azimuth_idx_list=azimuth_idx_list, class_name_list=class_name_list)
                        print(f"  cube vis time: {time.time() - time_start_vis}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1.0-mini")
    parser.add_argument("--data_dir", type=str, default="./dataset/nuscenes")
    parser.add_argument("--time_steps", type=int, default=2)
    parser.add_argument("--vis", default=False, action='store_true', help="Vis cube.")

    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./models/icfar-net.pth')
    parser.add_argument('--save_dir', default='./sim_output/Sim_NuScenes_v1_mini')
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')

    # ICFARNet Settings    
    parser.add_argument('--hidden_dims', type=int, default=32, help="hidden_dims.")
    parser.add_argument('--output_dims', type=int, default=1, help="output_dims.")
    parser.add_argument('--attribute', action='store_true', help="attribute embedding or not")
    args = parser.parse_args()

    main(args)
