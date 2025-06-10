import torch.optim

import util.loader as loader
import argparse
import os
import sys
import numpy as np
from engine.launch import launch
from model.model import RADDet
from utils.collect_env import collect_env_info
from utils.dist_utils import get_rank
from dataset.radar_dataset_SA_Radar import RararDataset, CarradaDataset
from torch.utils.data import DataLoader
from model.yolo_head import decodeYolo, yoloheadToPredictions, nms
from model.yolo_loss import RadDetLoss, nms2DOverClass
from metrics import mAP
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import json

def main(args):
    # initialization
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    env_str = collect_env_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(env_str)

    config = loader.readConfig(config_file_name=args.config_dir)
    config_data = config["DATA"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]
    config_eval = config["EVALUATE"]

    if args.test_set_dir != "":
        config_data["test_set_dir"] = args.test_set_dir

    # load anchor boxes with order
    anchor_boxes = loader.readAnchorBoxes(anchor_boxes_file="./anchors.txt")
    num_classes = len(config_data["all_classes"])

    ### NOTE: using the yolo head shape out from model for data generator ###
    if args.model_type == "RADDet":
        model = RADDet(config_model, config_data, config_train, anchor_boxes)
    print(f"Load pretrained model from {args.resume_from}")
    model.load_state_dict(torch.load(args.resume_from))
    model.to(device)

    model.eval()

    dataset_list = []
    if args.dataset == 'RADDET':
        dataset_list = ['raddet']
    elif args.dataset == 'CARRADA':
        dataset_list = ['carrada']

    for dataset_name in dataset_list:
        if dataset_name == 'raddet':
            test_dataset = RararDataset(config_data, config_train, config_model,
                                        config_model["feature_out_shape"], anchor_boxes, dType="test")
        elif dataset_name == 'carrada':
            test_dataset = CarradaDataset(config_data, config_train, config_model,
                                        config_model["feature_out_shape"], anchor_boxes, dType="test")

    data_dict = {}
    for idx, rad_dir in enumerate(test_dataset.RAD_sequences_test):
        batch_idx = idx // config_train["batch_size"]
        itr_idx= idx % config_train["batch_size"]
        data_dict[f'{batch_idx}_{itr_idx}'] = rad_dir
        
    # test_dataset.RAD_sequences_test
    with open('data.json', 'w') as json_file:
        json.dump(data_dict, json_file)

    test_loader = DataLoader(test_dataset,
                             batch_size=1, #config_train["batch_size"]//args.num_gpus, # 1, #
                             shuffle=False,
                             num_workers=1, #4,
                             pin_memory=True,
                             persistent_workers=True)
    if get_rank() == 0:
        ### NOTE: training settings ###
        logdir = os.path.join(config_train["log_dir"],
                              "b_" + str(config_train["batch_size"]) + "lr_" + str(config_train["learningrate_init"]))
        if not os.path.exists(logdir):
            os.makedirs(logdir)

    anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)
    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)

    criterion = RadDetLoss(
        input_size=input_size,
        focal_loss_iou_threshold=config_train["focal_loss_iou_threshold"]
    )

    print(f"start validation")
    mean_ap_test_list = [0.0, 0.0, 0.0, 0.0]
    ap_all_class_test_list = [[],[],[],[]]
    ap_all_class_list = [[],[],[],[]]
    for ap_all_class in ap_all_class_list:
        for class_id in range(num_classes):
            ap_all_class.append([])

    mean_ap_test_list_RA = [0.0, 0.0, 0.0, 0.0]
    ap_all_class_test_list_RA = [[],[],[],[]]
    ap_all_class_list_RA = [[],[],[],[]]
    for ap_all_class in ap_all_class_list_RA:
        for class_id in range(num_classes):
            ap_all_class.append([])

    total_losstest = []
    box_losstest = []
    conf_losstest = []
    category_losstest = []
    for idx, d in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            data, label, raw_boxes = d
            data = data.to(device)
            label = label.to(device)
            raw_boxes = raw_boxes.to(device) # (B, 30, 7), 7: box_xyzwhd+class_id
            _, feature = model(data)
            pred_raw, pred = decodeYolo(feature,
                                        input_size=input_size,
                                        anchor_boxes=anchor_boxes,
                                        scale=config_model["yolohead_xyz_scales"][0])
            box_loss, conf_loss, category_loss = criterion(pred_raw, pred, label, raw_boxes[..., :6])
            box_loss_b, conf_loss_b, category_loss_b = box_loss.cpu().detach(), conf_loss.cpu().detach(), \
                category_loss.cpu().detach()
            total_losstest.append(box_loss_b+conf_loss_b+category_loss_b)
            box_losstest.append(box_loss_b)
            conf_losstest.append(conf_loss_b)
            category_losstest.append(category_loss_b)
            raw_boxes = raw_boxes.cpu().numpy()
            pred = pred.cpu().detach().numpy()
            for batch_id in range(raw_boxes.shape[0]):
                raw_boxes_frame = raw_boxes[batch_id] # (30, 7)
                pred_frame = pred[batch_id] # -1: [x, y, z, w, h, d, score, p_class1, p_class2, ..., p_class6]
                pred_ori_information = pred_frame.reshape(-1, pred_frame.shape[-1])
                predicitons = yoloheadToPredictions(pred_frame, conf_threshold=config_model["confidence_threshold"]) # (N, 8), [x, y, z, w, h, d, score, class_index]
                nms_pred = nms(predicitons, config_model["nms_iou3d_threshold"],
                               config_model["input_shape"], sigma=0.3, method="nms")
                for idx_iou, iou_map in enumerate(config_eval["mAP_iou3d_threshold"]):
                    mean_ap, ap_all_class_list[idx_iou] = mAP.mAP(nms_pred, raw_boxes_frame,
                                                    config_model["input_shape"], ap_all_class_list[idx_iou],
                                                    tp_iou_threshold=iou_map)
                    mean_ap_test_list[idx_iou] += mean_ap

                predicitons_RA = np.zeros((predicitons.shape[0], 6))
                predicitons_RA[:, 0:2] = predicitons[:, 0:2]
                predicitons_RA[:, 2:4] = predicitons[:, 3:5]
                predicitons_RA[:, 4:6] = predicitons[:, 6:8]
                nms_pred_RA = nms2DOverClass(predicitons_RA, config_model["nms_iou3d_threshold"],
                                config_model["input_shape"], sigma=0.3, method="nms")
                raw_boxes_frame_RA = np.zeros((raw_boxes_frame.shape[0], 5))
                raw_boxes_frame_RA[:, 0:2] = raw_boxes_frame[:, 0:2]
                raw_boxes_frame_RA[:, 2:4] = raw_boxes_frame[:, 3:5]
                raw_boxes_frame_RA[:, 4] = raw_boxes_frame[:, 6]
                for idx_iou, iou_map in enumerate(config_eval["mAP_iou3d_threshold"]):
                    mean_ap, ap_all_class_list_RA[idx_iou] = mAP.mAP2D(nms_pred_RA, raw_boxes_frame_RA,
                                                    config_model["input_shape"], ap_all_class_list_RA[idx_iou],
                                                    tp_iou_threshold=iou_map)
                    mean_ap_test_list_RA[idx_iou] += mean_ap

    for idx_ap, ap_all_class in enumerate(ap_all_class_list):
        for ap_class_i in ap_all_class:
            if len(ap_class_i) == 0:
                class_ap = 0.
            else:
                class_ap = np.mean(ap_class_i)
            ap_all_class_test_list[idx_ap].append(class_ap)

    for idx_ap, ap_all_class in enumerate(ap_all_class_list_RA):
        for ap_class_i in ap_all_class:
            if len(ap_class_i) == 0:
                class_ap = 0.
            else:
                class_ap = np.mean(ap_class_i)
            ap_all_class_test_list_RA[idx_ap].append(class_ap)

    for mAP_iou3d, mean_ap_test, ap_all_class_test in zip(config_eval["mAP_iou3d_threshold"], mean_ap_test_list, ap_all_class_test_list):
        mean_ap_test /= len(test_dataset)
        print("-------> mAP_iou: %.3f, ap_all: %.6f, ap_person: %.6f, ap_bicycle: %.6f, ap_car: %.6f, ap_motorcycle: %.6f, ap_bus: %.6f, "
            "ap_truck: %.6f" % (mAP_iou3d, mean_ap_test, ap_all_class_test[0], ap_all_class_test[1], ap_all_class_test[2],
                                ap_all_class_test[3], ap_all_class_test[4], ap_all_class_test[5]))

    for mAP_iou3d, mean_ap_test, ap_all_class_test in zip(config_eval["mAP_iou3d_threshold"], mean_ap_test_list_RA, ap_all_class_test_list_RA):
        mean_ap_test /= len(test_dataset)
        print("-------> mAP_iou_RA: %.3f, ap_all: %.6f, ap_person: %.6f, ap_bicycle: %.6f, ap_car: %.6f, ap_motorcycle: %.6f, ap_bus: %.6f, "
            "ap_truck: %.6f" % (mAP_iou3d, mean_ap_test, ap_all_class_test[0], ap_all_class_test[1], ap_all_class_test[2],
                                ap_all_class_test[3], ap_all_class_test[4], ap_all_class_test[5]))

    print("-------> total_loss: %.6f, box_loss: %.6f, conf_loss: %.6f, category_loss: %.6f" %
        (np.mean(total_losstest), np.mean(box_losstest), np.mean(conf_losstest), np.mean(category_losstest)))


def get_parse():
    parser = argparse.ArgumentParser(description='Args for segmentation model.')
    parser.add_argument("--config_dir", type=str, default="./configs/config.json")
    parser.add_argument("--model_type", type=str, default="RADDet")
    parser.add_argument("--test_set_dir", type=str, default="")
    parser.add_argument("--dataset", type=str, default="RADDET")
    parser.add_argument("--num-gpus", type=int,
                        default=1,
                        help="Inference code only support single GPU.")
    parser.add_argument("--num-machines", type=int,
                        default=1,
                        help="The number of machines.")
    parser.add_argument("--machine-rank", type=int,
                        default=0,
                        help="The rank of current machine.")
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--dist_url", type=str,
                        default="tcp://127.0.0.1:{}".format(port),
                        help="initialization URL for pytorch distributed backend.")
    parser.add_argument("--resume_from", type=str,
                        default="/home/albert_wei/WorkSpaces_2023/RADIA/RADDet/RADDet_Pytorch/logs/RadarResNet/b_4lr_0.0001/ckpt/best.pth",
                        help="The number of machines.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parse()
    print("Command Line Args: ", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )