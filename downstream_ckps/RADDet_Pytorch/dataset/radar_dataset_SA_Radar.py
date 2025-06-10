from torch.utils.data import Dataset
import numpy as np
import os, glob
import util.loader as loader
import util.helper as helper
from torchvision.transforms import ToTensor
import torch
import pdb
import json

def gtfileFromRADfile_v2(RAD_file):
    """ Transfer RAD filename to gt filename """
    RAD_file_spec = RAD_file.split("RAD")[-1]
    prefix = RAD_file.split("/RAD/")[-2]
    gt_file = os.path.join(prefix, "gt") + RAD_file_spec.replace("npy", "pickle")
    return gt_file

class RararDataset(Dataset):
    def __init__(self, config_data, config_train, config_model, headoutput_shape,
                 anchors, transformer=ToTensor(), anchors_cart=None, cart_shape=None, dType="train", RADDir="RAD", add_train_dir=[], only_sim=False, real_percent=1.0):
        super(RararDataset, self).__init__()
        self.input_size = config_model["input_shape"]
        self.config_data = config_data
        self.config_train = config_train
        self.config_model = config_model
        self.headoutput_shape = headoutput_shape
        self.cart_shape = cart_shape
        self.grid_strides = self.getGridStrides()
        self.cart_grid_strides = self.getCartGridStrides()
        self.anchor_boxes = anchors
        self.anchor_boxes_cart = anchors_cart
        self.RADDir = RADDir
        self.RAD_sequences_train = self.readaddSequences(self.config_data["train_set_dir"]) # self.readSequences(mode="train")
        self.RAD_sequences_test = self.readaddSequences(self.config_data["test_set_dir"])# self.readSequences(mode="test")
        ### NOTE: if "if_validat" set true in "config.json", it will split trainset ###
        self.RAD_sequences_train, self.RAD_sequences_validate = self.splitTrain(self.RAD_sequences_train)
        self.RAD_sequences_train = self.RAD_sequences_train[:int(real_percent * len(self.RAD_sequences_train))]
        self.batch_size = config_train["batch_size"]
        self.total_train_batches = (self.config_train["epochs"] * len(self.RAD_sequences_train)) // self.batch_size
        self.total_test_batches = len(self.RAD_sequences_test) // self.batch_size
        self.total_validate_batches = len(self.RAD_sequences_validate) // self.batch_size
        self.dtype = dType
        self.transform = transformer
        # self.add_train_dir = add_train_dir
        if only_sim:
            for idx, add_dir in enumerate(add_train_dir):
                if idx == 0:
                    self.RAD_sequences_train = self.readaddSequences(add_dir)
                else:
                    self.RAD_sequences_train += self.readaddSequences(add_dir)
        else:
            for idx, add_dir in enumerate(add_train_dir):
                if add_dir == 'none':
                    continue
                self.RAD_sequences_train += self.readaddSequences(add_dir)

    def __len__(self):
        if self.dtype == "train":
            return len(self.RAD_sequences_train)
        elif self.dtype == "validate":
            return len(self.RAD_sequences_validate)
        elif self.dtype == "test":
            return len(self.RAD_sequences_test)
        else:
            raise ValueError("This type of dataset does not exist.")

    def __getitem__(self, index):
        if self.dtype == "train":
            return self.trainData(index)
        elif self.dtype == "validate":
            return self.valData(index)
        elif self.dtype == "test":
            return self.testData(index)
        else:
            raise ValueError("This type of dataset does not exist.")

    def trainData(self, index):
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_train[index]
            RAD_complex = loader.readRAD(RAD_filename) # (256,256,64)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            if RAD_complex.dtype not in ["float16", "float32", "float64"]:
                RAD_data = helper.complexTo2Channels(RAD_complex)
            else:
                RAD_data = RAD_complex.astype("float32")
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                       self.config_data["global_variance_log"]
            ### load ground truth instances ###
            gt_filename = gtfileFromRADfile_v2(RAD_filename)
            gt_instances = loader.readRadarInstances(gt_filename) #.pikcle -> class, boxes, cart_boxes
            if gt_instances is None:
                print(gt_filename)
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)
            index += 1
            gt_labels = np.stack(gt_labels, axis=0) # (16, 16, 4, 6, 13)
            if has_label:
                if self.transform:
                    return self.transform(RAD_data), torch.tensor(gt_labels, dtype=torch.float32), \
                        torch.tensor(raw_boxes, dtype=torch.float32)
                else:
                    return RAD_data, gt_labels, raw_boxes

    def valData(self, index):
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_validate[index]
            RAD_complex = loader.readRAD(RAD_filename) # (256,256,64)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            if RAD_complex.dtype not in ["float16", "float32", "float64"]:
                RAD_data = helper.complexTo2Channels(RAD_complex)
            else:
                RAD_data = RAD_complex.astype("float32")
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                       self.config_data["global_variance_log"]
            ### load ground truth instances ###
            gt_filename = gtfileFromRADfile_v2(RAD_filename)
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)
            index += 1
            gt_labels = np.stack(gt_labels, axis=0)
            if has_label:
                if self.transform:
                    return self.transform(RAD_data), torch.tensor(gt_labels, dtype=torch.float32), \
                        torch.tensor(raw_boxes, dtype=torch.float32)
                else:
                    return RAD_data, gt_labels, raw_boxes

    def testData(self, index):
        """ Generate test data with batch size """
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_test[index]
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            # print(RAD_complex.dtype)
            if RAD_complex.dtype not in ["float16", "float32", "float64"]:
                RAD_data = helper.complexTo2Channels(RAD_complex)
            else:
                RAD_data = RAD_complex.astype("float32")
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / self.config_data["global_variance_log"]
            ### load ground truth instances ###
            gt_filename = gtfileFromRADfile_v2(RAD_filename)
            gt_instances = loader.readRadarInstances(gt_filename)
            if gt_instances is None:
                raise ValueError("gt file not found, please double check the path")

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)
            index += 1
            gt_labels = np.stack(gt_labels, axis=0)
            if has_label:
                if self.transform:
                    return self.transform(RAD_data), torch.tensor(gt_labels, dtype=torch.float32), \
                        torch.tensor(raw_boxes, dtype=torch.float32)
                else:
                    return RAD_data, gt_labels, raw_boxes

    def getGridStrides(self, ):
        """ Get grid strides """
        strides = (np.array(self.config_model["input_shape"])[:3] / np.array(self.headoutput_shape[1:4]))
        return np.array(strides).astype(np.float32)

    def getCartGridStrides(self, ):
        """ Get grid strides """
        if self.cart_shape is not None:
            cart_output_shape = [int(self.config_model["input_shape"][0]),
                                 int(2 * self.config_model["input_shape"][0])]
            strides = (np.array(cart_output_shape) / np.array(self.cart_shape[1:3]))
            return np.array(strides).astype(np.float32)
        else:
            return None

    def readSequences(self, mode):
        """ Read sequences from train/test directories. """
        assert mode in ["train", "test"]
        if mode == "train":
            sequences = glob.glob(os.path.join(self.config_data["train_set_dir"], f"{self.RADDir}/*/*.npy"))
        else:
            sequences = glob.glob(os.path.join(self.config_data["test_set_dir"], f"{self.RADDir}/*/*.npy"))
        if len(sequences) == 0:
            raise ValueError("Cannot read data from either train or test directory, "
                             "Please double-check the data path or the data format.")
        return sequences
    
    def readaddSequences(self, path):
        """ Read sequences from train/test directories. """
        sequences = glob.glob(f"{path}/*/*.npy") + glob.glob(f"{path}/*/*/*.npy")
        sequences = sorted(sequences)
        if len(sequences) == 0:
            raise ValueError("Cannot read data from either train or test directory, "
                             "Please double-check the data path or the data format.")
        return sequences

    def splitTrain(self, train_sequences):
        """ Split train set to train and validate """
        total_num = len(train_sequences)
        validate_num = int(0.1 * total_num)
        if self.config_train["if_validate"]:
            return train_sequences[:total_num-validate_num], \
                train_sequences[total_num-validate_num:]
        else:
            return train_sequences, train_sequences[total_num-validate_num:]

    def encodeToLabels(self, gt_instances):
        """ Transfer ground truth instances into Detection Head format """
        raw_boxes_xyzwhd = np.zeros((self.config_data["max_boxes_per_frame"], 7))
        ### initialize gronud truth labels as np.zeors ###
        gt_labels = np.zeros(list(self.headoutput_shape[1:4]) + \
                             [len(self.anchor_boxes)] + \
                             [len(self.config_data["all_classes"]) + 7])
        # (16, 16, 4, 6, 13)
        ### start transferring box to ground turth label format ###
        for i in range(len(gt_instances["classes"])):
            if i > self.config_data["max_boxes_per_frame"]:
                continue
            class_name = gt_instances["classes"][i]
            box_xyzwhd = gt_instances["boxes"][i]
            class_id = self.config_data["all_classes"].index(class_name)
            if i < self.config_data["max_boxes_per_frame"]:
                raw_boxes_xyzwhd[i, :6] = box_xyzwhd
                raw_boxes_xyzwhd[i, 6] = class_id
            class_onehot = helper.smoothOnehot(class_id, len(self.config_data["all_classes"]))

            exist_positive = False

            grid_strid = self.grid_strides
            anchor_stage = self.anchor_boxes # (6, 3)
            box_xyzwhd_scaled = box_xyzwhd[np.newaxis, :].astype(np.float32) # broadcast (1,6)
            box_xyzwhd_scaled[:, :3] /= grid_strid
            anchorstage_xyzwhd = np.zeros([len(anchor_stage), 6])
            anchorstage_xyzwhd[:, :3] = np.floor(box_xyzwhd_scaled[:, :3]) + 0.5
            anchorstage_xyzwhd[:, 3:] = anchor_stage.astype(np.float32)

            iou_scaled = helper.iou3d(box_xyzwhd_scaled, anchorstage_xyzwhd, \
                                      self.input_size)
            ### NOTE: 0.3 is from YOLOv4, maybe this should be different here ###
            ### it means, as long as iou is over 0.3 with an anchor, the anchor
            ### should be taken into consideration as a ground truth label
            iou_mask = iou_scaled > 0.3

            if np.any(iou_mask):
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]). \
                    astype(np.int32)
                ### TODO: consider changing the box to raw yolohead output format ###
                gt_labels[xind, yind, zind, iou_mask, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, iou_mask, 6:7] = 1.
                gt_labels[xind, yind, zind, iou_mask, 7:] = class_onehot
                exist_positive = True

            if not exist_positive:
                ### NOTE: this is the normal one ###
                ### it means take the anchor box with maximum iou to the raw
                ### box as the ground truth label
                anchor_ind = np.argmax(iou_scaled)
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]). \
                    astype(np.int32)
                gt_labels[xind, yind, zind, anchor_ind, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, anchor_ind, 6:7] = 1.
                gt_labels[xind, yind, zind, anchor_ind, 7:] = class_onehot

        has_label = False
        for label_stage in gt_labels:
            if label_stage.max() != 0:
                has_label = True
        gt_labels = [np.where(gt_i == 0, 1e-16, gt_i) for gt_i in gt_labels]
        return gt_labels, has_label, raw_boxes_xyzwhd

class CarradaDataset(Dataset):
    def __init__(self, config_data, config_train, config_model, headoutput_shape,
                 anchors, transformer=ToTensor(), anchors_cart=None, cart_shape=None, dType="train", RADDir="RAD", add_train_dir=[], only_sim=False, real_percent=1.0, new_anno=False):
        super(CarradaDataset, self).__init__()
        self.input_size = config_model["input_shape"]
        self.config_data = config_data
        self.config_train = config_train
        self.config_model = config_model
        self.headoutput_shape = headoutput_shape
        self.cart_shape = cart_shape
        self.grid_strides = self.getGridStrides()
        self.cart_grid_strides = self.getCartGridStrides()
        self.anchor_boxes = anchors
        self.anchor_boxes_cart = anchors_cart
        self.RADDir = RADDir
        self.new_anno = new_anno
        self.RAD_sequences_train, self.box_list_train = self.readaddSequences(self.config_data["train_set_dir"], 'train') # self.readSequences(mode="train")
        self.RAD_sequences_test, self.box_list_test = self.readaddSequences(self.config_data["test_set_dir"], 'test')# self.readSequences(mode="test")
        ### NOTE: if "if_validat" set true in "config.json", it will split trainset ###
        self.RAD_sequences_train, self.RAD_sequences_validate, self.box_list_train, self.box_list_validate = self.splitTrain(self.RAD_sequences_train, self.box_list_train)
        self.RAD_sequences_train = self.RAD_sequences_train[:int(real_percent * len(self.RAD_sequences_train))]
        self.box_list_train = self.box_list_train[:int(real_percent * len(self.box_list_train))]
        self.batch_size = config_train["batch_size"]
        self.total_train_batches = (self.config_train["epochs"] * len(self.RAD_sequences_train)) // self.batch_size
        self.total_test_batches = len(self.RAD_sequences_test) // self.batch_size
        self.total_validate_batches = len(self.RAD_sequences_validate) // self.batch_size
        self.dtype = dType
        self.transform = transformer
        # self.add_train_dir = add_train_dir
        if only_sim:
            for idx, add_dir in enumerate(add_train_dir):
                if idx == 0:
                    self.RAD_sequences_train, self.box_list_train = self.readaddSequences(add_dir, 'train')
                else:
                    RAD_sequences, box_list = self.readaddSequences(add_dir, 'train')
                    self.RAD_sequences_train += RAD_sequences
                    self.box_list_train += box_list
        else:
            for idx, add_dir in enumerate(add_train_dir):
                if add_dir == 'none':
                    continue
                RAD_sequences, box_list = self.readaddSequences(add_dir, 'train')
                self.RAD_sequences_train += RAD_sequences
                self.box_list_train += box_list

    def __len__(self):
        if self.dtype == "train":
            return len(self.RAD_sequences_train)
        elif self.dtype == "validate":
            return len(self.RAD_sequences_validate)
        elif self.dtype == "test":
            return len(self.RAD_sequences_test)
        else:
            raise ValueError("This type of dataset does not exist.")

    def __getitem__(self, index):
        if self.dtype == "train":
            return self.trainData(index)
        elif self.dtype == "validate":
            return self.valData(index)
        elif self.dtype == "test":
            return self.testData(index)
        else:
            raise ValueError("This type of dataset does not exist.")

    def trainData(self, index):
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_train[index]
            RAD_complex = loader.readRAD(RAD_filename) # (256,256,64)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            if 'sim_data' not in RAD_filename:
                RAD_data = pow(RAD_complex, 2)
                RAD_data = np.log10(RAD_data + 1.)
                RAD_data = RAD_data.astype("float32")
            else:
                RAD_data = RAD_complex.astype("float32")
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                       self.config_data["global_variance_log"]
            ### load ground truth instances ###
            box_list = self.box_list_train[index]
            gt_instances = {"classes":[], "boxes":[]}
            class_list = ["motorcycle", "person", "bicycle", "car"]
            for box in box_list:
                [x_begin, x_end, y_begin, y_end, z_begin, z_end, label] = box # r, a, d
                box_xyzwhd = np.array([(x_begin+x_end)/2, (y_begin+y_end)/2, (z_begin+z_end)/2, x_end-x_begin, y_end-y_begin, z_end-z_begin])
                class_name = class_list[int(label)]
                gt_instances["classes"].append(class_name)
                gt_instances["boxes"].append(box_xyzwhd)

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)
            index += 1
            gt_labels = np.stack(gt_labels, axis=0) # (16, 16, 4, 6, 13)
            if has_label:
                if self.transform:
                    return self.transform(RAD_data), torch.tensor(gt_labels, dtype=torch.float32), \
                        torch.tensor(raw_boxes, dtype=torch.float32)
                else:
                    return RAD_data, gt_labels, raw_boxes

    def valData(self, index):
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_validate[index]
            RAD_complex = loader.readRAD(RAD_filename) # (256,256,64)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            if 'sim_data' not in RAD_filename:
                RAD_data = pow(RAD_complex, 2)
                RAD_data = np.log10(RAD_data + 1.)
                RAD_data = RAD_data.astype("float32")
            else:
                RAD_data = RAD_complex.astype("float32")
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / \
                       self.config_data["global_variance_log"]
            ### load ground truth instances ###
            box_list = self.box_list_validate[index]
            gt_instances = {"classes":[], "boxes":[]}
            class_list = ["motorcycle", "person", "bicycle", "car"]
            for box in box_list:
                [x_begin, x_end, y_begin, y_end, z_begin, z_end, label] = box # r, a, d
                box_xyzwhd = np.array([(x_begin+x_end)/2, (y_begin+y_end)/2, (z_begin+z_end)/2, x_end-x_begin, y_end-y_begin, z_end-z_begin])
                class_name = class_list[int(label)]
                gt_instances["classes"].append(class_name)
                gt_instances["boxes"].append(box_xyzwhd)

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)
            index += 1
            gt_labels = np.stack(gt_labels, axis=0)
            if has_label:
                if self.transform:
                    return self.transform(RAD_data), torch.tensor(gt_labels, dtype=torch.float32), \
                        torch.tensor(raw_boxes, dtype=torch.float32)
                else:
                    return RAD_data, gt_labels, raw_boxes

    def testData(self, index):
        """ Generate test data with batch size """
        has_label = False
        while not has_label:
            RAD_filename = self.RAD_sequences_test[index]
            RAD_complex = loader.readRAD(RAD_filename)
            if RAD_complex is None:
                raise ValueError("RAD file not found, please double check the path")
            ### NOTE: Gloabl Normalization ###
            if 'sim_data' not in RAD_filename:
                RAD_data = pow(RAD_complex, 2)
                RAD_data = np.log10(RAD_data + 1.)
                RAD_data = RAD_data.astype("float32")
            else:
                RAD_data = RAD_complex.astype("float32")
            RAD_data = (RAD_data - self.config_data["global_mean_log"]) / self.config_data["global_variance_log"]
            ### load ground truth instances ###
            box_list = self.box_list_test[index]
            gt_instances = {"classes":[], "boxes":[]}
            class_list = ["motorcycle", "person", "bicycle", "car"]
            for box in box_list:
                [x_begin, x_end, y_begin, y_end, z_begin, z_end, label] = box # r, a, d
                box_xyzwhd = np.array([(x_begin+x_end)/2, (y_begin+y_end)/2, (z_begin+z_end)/2, x_end-x_begin, y_end-y_begin, z_end-z_begin])
                class_name = class_list[int(label)]
                gt_instances["classes"].append(class_name)
                gt_instances["boxes"].append(box_xyzwhd)

            ### NOTE: decode ground truth boxes to YOLO format ###
            gt_labels, has_label, raw_boxes = self.encodeToLabels(gt_instances)
            index += 1
            gt_labels = np.stack(gt_labels, axis=0)
            if has_label:
                if self.transform:
                    return self.transform(RAD_data), torch.tensor(gt_labels, dtype=torch.float32), \
                        torch.tensor(raw_boxes, dtype=torch.float32)
                else:
                    return RAD_data, gt_labels, raw_boxes

    def getGridStrides(self, ):
        """ Get grid strides """
        strides = (np.array(self.config_model["input_shape"])[:3] / np.array(self.headoutput_shape[1:4]))
        return np.array(strides).astype(np.float32)

    def getCartGridStrides(self, ):
        """ Get grid strides """
        if self.cart_shape is not None:
            cart_output_shape = [int(self.config_model["input_shape"][0]),
                                 int(2 * self.config_model["input_shape"][0])]
            strides = (np.array(cart_output_shape) / np.array(self.cart_shape[1:3]))
            return np.array(strides).astype(np.float32)
        else:
            return None

    def readSequences(self, mode):
        """ Read sequences from train/test directories. """
        assert mode in ["train", "test"]
        if mode == "train":
            sequences = glob.glob(os.path.join(self.config_data["train_set_dir"], f"{self.RADDir}/*/*.npy"))
        else:
            sequences = glob.glob(os.path.join(self.config_data["test_set_dir"], f"{self.RADDir}/*/*.npy"))
        if len(sequences) == 0:
            raise ValueError("Cannot read data from either train or test directory, "
                             "Please double-check the data path or the data format.")
        return sequences
    
    def readaddSequences(self, path, split):
        """ Read sequences from train/test directories. """
        split_json_path = 'path/Carrada/Carrada/data_seq_ref.json'
        with open(split_json_path, 'r') as f:
            split_json = json.load(f)
        if split == 'train':
            split_list = ["Train", "Validation"]
        else:
            split_list = ["Test"]
        sequences_all = []
        for date_name, v in split_json.items():
            if v["split"] in split_list:
                sequences_all += sorted(glob.glob(os.path.join(path, f'datasets_master/*/{date_name}/*/*.npy')))

        sequences = []
        box_list = []
        for sequence in sequences_all:
            boxes = []
            date_name = sequence.split('/')[-3]
            rad_name = sequence.split('/')[-1].split('.')[0]

            label_ra_path = os.path.join('path/Carrada', f'Carrada/{date_name}/annotations/box/range_angle_light.json')
            label_rd_path = os.path.join('path/Carrada', f'Carrada/{date_name}/annotations/box/range_doppler_light.json')
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

            sequences.append(sequence)
            box_list.append(boxes)

        if len(sequences) == 0:
            raise ValueError("Cannot read data from either train or test directory, "
                             "Please double-check the data path or the data format.")
        else:
            print(f"Adding {len(sequences)} samples from {path} for {split}")
        return sequences, box_list

    def splitTrain(self, train_sequences, box_list_train):
        """ Split train set to train and validate """
        total_num = len(train_sequences)
        validate_num = int(0.1 * total_num)
        if self.config_train["if_validate"]:
            return train_sequences[:total_num-validate_num], \
                train_sequences[total_num-validate_num:], box_list_train[:total_num-validate_num], box_list_train[total_num-validate_num:]
        else:
            return train_sequences, train_sequences[total_num-validate_num:], box_list_train, box_list_train[total_num-validate_num:]

    def encodeToLabels(self, gt_instances):
        """ Transfer ground truth instances into Detection Head format """
        raw_boxes_xyzwhd = np.zeros((self.config_data["max_boxes_per_frame"], 7))
        ### initialize gronud truth labels as np.zeors ###
        gt_labels = np.zeros(list(self.headoutput_shape[1:4]) + \
                             [len(self.anchor_boxes)] + \
                             [len(self.config_data["all_classes"]) + 7])
        # (16, 16, 4, 6, 13)
        ### start transferring box to ground turth label format ###
        for i in range(len(gt_instances["classes"])):
            if i > self.config_data["max_boxes_per_frame"]:
                continue
            class_name = gt_instances["classes"][i]
            box_xyzwhd = gt_instances["boxes"][i]
            class_id = self.config_data["all_classes"].index(class_name)
            if i < self.config_data["max_boxes_per_frame"]:
                raw_boxes_xyzwhd[i, :6] = box_xyzwhd 
                raw_boxes_xyzwhd[i, 6] = class_id 
            class_onehot = helper.smoothOnehot(class_id, len(self.config_data["all_classes"]))

            exist_positive = False

            grid_strid = self.grid_strides
            anchor_stage = self.anchor_boxes # (6, 3)
            box_xyzwhd_scaled = box_xyzwhd[np.newaxis, :].astype(np.float32) # broadcast (1,6)
            box_xyzwhd_scaled[:, :3] /= grid_strid
            anchorstage_xyzwhd = np.zeros([len(anchor_stage), 6])
            anchorstage_xyzwhd[:, :3] = np.floor(box_xyzwhd_scaled[:, :3]) + 0.5
            anchorstage_xyzwhd[:, 3:] = anchor_stage.astype(np.float32)

            iou_scaled = helper.iou3d(box_xyzwhd_scaled, anchorstage_xyzwhd, \
                                      self.input_size)
            ### NOTE: 0.3 is from YOLOv4, maybe this should be different here ###
            ### it means, as long as iou is over 0.3 with an anchor, the anchor
            ### should be taken into consideration as a ground truth label
            iou_mask = iou_scaled > 0.3

            if np.any(iou_mask):
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]). \
                    astype(np.int32)
                ### TODO: consider changing the box to raw yolohead output format ###
                gt_labels[xind, yind, zind, iou_mask, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, iou_mask, 6:7] = 1.
                gt_labels[xind, yind, zind, iou_mask, 7:] = class_onehot
                exist_positive = True

            if not exist_positive:
                ### NOTE: this is the normal one ###
                ### it means take the anchor box with maximum iou to the raw
                ### box as the ground truth label
                anchor_ind = np.argmax(iou_scaled)
                xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]). \
                    astype(np.int32)
                gt_labels[xind, yind, zind, anchor_ind, 0:6] = box_xyzwhd
                gt_labels[xind, yind, zind, anchor_ind, 6:7] = 1.
                gt_labels[xind, yind, zind, anchor_ind, 7:] = class_onehot

        has_label = False
        for label_stage in gt_labels:
            if label_stage.max() != 0:
                has_label = True
        gt_labels = [np.where(gt_i == 0, 1e-16, gt_i) for gt_i in gt_labels]
        return gt_labels, has_label, raw_boxes_xyzwhd