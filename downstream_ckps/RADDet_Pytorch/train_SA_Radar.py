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
from tensorboardX import SummaryWriter
from model.yolo_head import decodeYolo, yoloheadToPredictions, nms
from model.yolo_loss import RadDetLoss
from utils.optimizer_utils import LinearWarmupCosineAnnealingLR
from metrics import mAP
import utils.dist_utils as dist_utils
import time
import pdb
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main(args):
    # initialization
    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    env_str = collect_env_info()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(env_str)

    config = loader.readConfig(config_file_name=args.config_dir)
    config_data = config["DATA"]
    config_radar = config["RADAR_CONFIGURATION"]
    config_model = config["MODEL"]
    config_train = config["TRAIN"]
    config_evaluate = config["EVALUATE"]

    # load anchor boxes with order
    anchor_boxes = loader.readAnchorBoxes(anchor_boxes_file="./anchors.txt")
    num_classes = len(config_data["all_classes"])

    ### NOTE: using the yolo head shape out from model for data generator ###
    if args.model_type == "RADDet":
        model = RADDet(config_model, config_data, config_train, anchor_boxes)

    model.to(device)

    dataset_list = []
    if args.dataset == 'RADDET':
        dataset_list = ['raddet']
    elif args.dataset == 'CARRADA':
        dataset_list = ['carrada']
    
    train_dataset, validate_dataset, test_dataset = None, None, None

    for dataset_name in dataset_list:
        if dataset_name == 'raddet':
            new_train_dataset = RararDataset(config_data, config_train, config_model,
                                        config_model["feature_out_shape"], anchor_boxes, dType="train", add_train_dir=args.add_train_dir, only_sim=args.only_sim, real_percent=args.real_percent)
                
            new_validate_dataset = RararDataset(config_data, config_train, config_model,
                                            config_model["feature_out_shape"], anchor_boxes, dType="validate")
            new_test_dataset = RararDataset(config_data, config_train, config_model,
                                        config_model["feature_out_shape"], anchor_boxes, dType="test")    # 2032
        elif dataset_name == 'carrada':
            new_train_dataset = CarradaDataset(config_data, config_train, config_model,
                                        config_model["feature_out_shape"], anchor_boxes, dType="train", add_train_dir=args.add_train_dir, only_sim=args.only_sim, real_percent=args.real_percent, new_anno=args.new_anno)

            new_validate_dataset = CarradaDataset(config_data, config_train, config_model,
                                            config_model["feature_out_shape"], anchor_boxes, dType="validate", new_anno=args.new_anno)
            new_test_dataset = CarradaDataset(config_data, config_train, config_model,
                                        config_model["feature_out_shape"], anchor_boxes, dType="test", new_anno=args.new_anno)    # 2032
            
        train_dataset = new_train_dataset if train_dataset is None else train_dataset + new_train_dataset
        validate_dataset = new_validate_dataset if validate_dataset is None else validate_dataset + new_validate_dataset
        test_dataset = new_test_dataset if test_dataset is None else test_dataset + new_test_dataset
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config_train["batch_size"]//args.num_gpus,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              persistent_workers=True)
    validate_loader = DataLoader(validate_dataset,
                                 batch_size=config_train["batch_size"]//args.num_gpus,
                                 shuffle=True,
                                 num_workers=4,
                                 pin_memory=True,
                                 persistent_workers=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=config_train["batch_size"]//args.num_gpus,
                             shuffle=False,
                             num_workers=4,
                             pin_memory=True,
                             persistent_workers=True)
    if get_rank() == 0:
        ### NOTE: training settings ###
        if args.only_sim:
            logdir = os.path.join(config_train["log_dir"],
                                config["NAME"] + "-only_sim-D_" + args.dataset + "-M_" + args.model_type + "-e_" + str(config_train["epochs"]) + "-b_" + str(config_train["batch_size"]) + "-lr_" + str(config_train["learningrate_init"])) + "-" + args.ps + "-len_" + str(len(train_loader)*config_train["batch_size"]//args.num_gpus)
        else:
            logdir = os.path.join(config_train["log_dir"],
                                config["NAME"] + "-D_" + args.dataset + "-M_" + args.model_type + "-e_" + str(config_train["epochs"]) + "-b_" + str(config_train["batch_size"]) + "-lr_" + str(config_train["learningrate_init"])) + "-" + args.ps + "-len_" + str(len(train_loader)*config_train["batch_size"]//args.num_gpus)

        if not os.path.exists(logdir):
            os.makedirs(logdir)
        writer = SummaryWriter(logdir=logdir)
        log_specific_dir = os.path.join(logdir, "ckpt")
        if not os.path.exists(log_specific_dir):
            os.makedirs(log_specific_dir)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config_train["learningrate_init"], betas=(0.9, 0.99))

    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                              warmup_epochs=config_train["warmup_steps"],
                                              eta_min=config_train["learningrate_end"],
                                              max_epochs=config_train["epochs"],
                                              warmup_start_lr=config_train["learningrate_init"])

    anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32).to(device)
    input_size = torch.tensor(list(config_model["input_shape"]), dtype=torch.float32).to(device)

    criterion = RadDetLoss(
        input_size=input_size,
        focal_loss_iou_threshold=config_train["focal_loss_iou_threshold"]
    )

    step, best_val = 0, 0.0
    total_epochs = config_train["epochs"]
    total_steps = len(train_loader) * total_epochs
    start_time = time.time()
    for i in range(config_train["epochs"]):
        for d in train_loader:
            data, label, raw_boxes = d
            data = data.to(device)
            label = label.to(device)
            raw_boxes = raw_boxes.to(device)
            # print(data.shape, label.shape, raw_boxes.shape, data.device)
            # # torch.Size([16, 64, 256, 256]) torch.Size([16, 16, 16, 4, 6, 13]) torch.Size([16, 30, 7]
            _, feature = model(data) # torch.Size([3, 64, 256, 256]) -> torch.Size([3, 256, 16, 16]) -> torch.Size([3, 512, 16, 16]) -> torch.Size([3, 4*78, 16, 16]) -> torch.Size([3, 4, 78, 16, 16]), (-1, int(self.yolo_feature_size[-1]/4), int(self.num_anchor) * (num_class + 7), in_feature_size[2:])
            pred_raw, pred = decodeYolo(feature,
                                        input_size=input_size,
                                        anchor_boxes=anchor_boxes,
                                        scale=config_model["yolohead_xyz_scales"][0])
            # pred_raw (16, 16, 16, 4, 6, 13)
            # raw_box (16, 30, 7)
            box_loss, conf_loss, category_loss = criterion(pred_raw, pred, label, raw_boxes[..., :6])
            total_loss = box_loss + conf_loss + category_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss_r, box_loss_r, conf_loss_r, category_loss_r = total_loss.cpu().detach(), \
                box_loss.cpu().detach(), conf_loss.cpu().detach(), category_loss.cpu().detach()

            if get_rank() == 0:
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=step)
                writer.add_scalar("loss/total_loss", total_loss_r, global_step=step)
                writer.add_scalar("loss/box_loss", box_loss_r, global_step=step)
                writer.add_scalar("loss/conf_loss", conf_loss_r, global_step=step)
                writer.add_scalar("loss/category_loss", category_loss_r, global_step=step)
                writer.flush()
            step += 1

            if get_rank() == 0:
                end_time = time.time()
                step_duration = end_time - start_time
                remaining_steps = total_steps - step
                remaining_time = remaining_steps * (step_duration / step)
                remaining_hours = int(remaining_time // 3600)
                remaining_minutes = int((remaining_time % 3600) // 60)
                print("===> epochs: %4d/%4d, train step: %4d/%4d, time: %02d h %02d min, lr: %.7f, total_loss: %4.2f, "
                      "box_loss: %4.2f, conf_loss: %4.2f, category_loss: %4.2f" %
                      (i, config_train["epochs"], step%len(train_loader), len(train_loader), remaining_hours, remaining_minutes, optimizer.param_groups[0]['lr'], total_loss_r, box_loss_r, conf_loss_r, category_loss_r))
        if get_rank() == 0:
            print(f"epochs: {i}, start validation")
            mean_ap_test = 0.0
            ap_all_class_test = []
            ap_all_class = []
            total_losstest = []
            box_losstest = []
            conf_losstest = []
            category_losstest = []
            for class_id in range(num_classes):
                ap_all_class.append([])
            model.eval()
            if config_train["if_validate"]:
                validate_loader = validate_loader
            else:
                validate_loader = test_loader
            for d in validate_loader:
                with torch.no_grad():
                    data, label, raw_boxes = d  #raw_boxed.shape = [16,30,7]
                    data = data.to(device)
                    label = label.to(device)
                    raw_boxes = raw_boxes.to(device)
                    feature_bk, feature = model(data)
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
                        raw_boxes_frame = raw_boxes[batch_id]
                        pred_frame = pred[batch_id]
                        predicitons = yoloheadToPredictions(pred_frame,
                                                            conf_threshold=config_model["confidence_threshold"])
                        nms_pred = nms(predicitons, config_model["nms_iou3d_threshold"],
                                       config_model["input_shape"], sigma=0.3, method="nms")
                        mean_ap, ap_all_class = mAP.mAP(nms_pred, raw_boxes_frame,
                                                        config_model["input_shape"], ap_all_class,
                                                        tp_iou_threshold=config_model["mAP_iou3d_threshold"])
                        mean_ap_test += mean_ap
            for ap_class_i in ap_all_class:
                if len(ap_class_i) == 0:
                    class_ap = 0.
                else:
                    class_ap = np.mean(ap_class_i)
                ap_all_class_test.append(class_ap)
            mean_ap_test /= len(validate_loader)
            print("-------> ap: %.6f" % mean_ap_test)

            if config_train["if_validate"]:
                writer.add_scalar("ap/ap_all", mean_ap_test, global_step=i)
                writer.add_scalar("ap/ap_person", ap_all_class_test[0], global_step=i)
                writer.add_scalar("ap/ap_bicycle", ap_all_class_test[1], global_step=i)
                writer.add_scalar("ap/ap_car", ap_all_class_test[2], global_step=i)
                writer.add_scalar("ap/ap_motorcycle", ap_all_class_test[3], global_step=i)
                writer.add_scalar("ap/ap_bus", ap_all_class_test[4], global_step=i)
                writer.add_scalar("ap/ap_truck", ap_all_class_test[5], global_step=i)
                ### NOTE: validate loss ###
                writer.add_scalar("validate_loss/total_loss",
                                np.mean(total_losstest), global_step=i)
                writer.add_scalar("validate_loss/box_loss",
                                np.mean(box_losstest), global_step=i)
                writer.add_scalar("validate_loss/conf_loss",
                                np.mean(conf_losstest), global_step=i)
                writer.add_scalar("validate_loss/category_loss",
                                np.mean(category_losstest), global_step=i)
                writer.flush()
            else:
                writer.add_scalar("ap_test/ap_all", mean_ap_test, global_step=i)
                writer.add_scalar("ap_test/ap_person", ap_all_class_test[0], global_step=i)
                writer.add_scalar("ap_test/ap_bicycle", ap_all_class_test[1], global_step=i)
                writer.add_scalar("ap_test/ap_car", ap_all_class_test[2], global_step=i)
                writer.add_scalar("ap_test/ap_motorcycle", ap_all_class_test[3], global_step=i)
                writer.add_scalar("ap_test/ap_bus", ap_all_class_test[4], global_step=i)
                writer.add_scalar("ap_test/ap_truck", ap_all_class_test[5], global_step=i)
                ### NOTE: validate loss ###
                writer.add_scalar("test_loss/total_loss",
                                np.mean(total_losstest), global_step=i)
                writer.add_scalar("test_loss/box_loss",
                                np.mean(box_losstest), global_step=i)
                writer.add_scalar("test_loss/conf_loss",
                                np.mean(conf_losstest), global_step=i)
                writer.add_scalar("test_loss/category_loss",
                                np.mean(category_losstest), global_step=i)
                writer.flush()

            ckpt_file_path = os.path.join(log_specific_dir, f"{i}.pth")
            ckpt_file_best_path = os.path.join(log_specific_dir, "best.pth")
            save_fre = config_train["epochs"] // 10
            if i%save_fre == save_fre-1:
                with open(ckpt_file_path, 'wb') as f:
                    torch.save(model.state_dict(), f)
                print("Saving checkpoint to {}".format(ckpt_file_path))
            if mean_ap_test > best_val:
                best_val = mean_ap_test
                with open(ckpt_file_best_path, 'wb') as f:
                    torch.save(model.state_dict(), f)
            print("Current model is the best model at present, and it is saved to {}".format(ckpt_file_best_path))
        if dist_utils.get_world_size() > 1 and dist_utils.get_world_size() > 0:
            torch.distributed.barrier()
        model.train()
        scheduler.step()


def get_parse():
    parser = argparse.ArgumentParser(description='Args for segmentation model.')
    parser.add_argument("--model_type", type=str, default="RADDet")
    parser.add_argument("--dataset", type=str, default="RADDET")
    parser.add_argument("--config_dir", type=str, default="./configs/config.json")
    parser.add_argument("--add_train_dir", nargs='+', default=['none'], help="add training datasets.") # , type=str, default=None)
    parser.add_argument("--only_sim", action="store_true", help="")
    parser.add_argument("--new_anno", action="store_true", help="")
    parser.add_argument("--real_percent", type=float, default=1.0)
    parser.add_argument("--ps", type=str, default='')
    parser.add_argument("--num-gpus", type=int,
                        default=1,
                        help="The number of gpus.")
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
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--resume", action="store_true", help="resume from previous ckpt file")
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