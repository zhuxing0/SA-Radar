import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.ICFARNet import ICFARNet, autocast
import core.datasets as datasets
from core.utils.utils import InputPadder
from PIL import Image
import torch.nn.functional as F
import pdb

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate(model, dataset=['raddet'], mixed_prec=False):
    """ Peform validation using the raddet dataset """
    model.eval()
    aug_params = {}

    val_dataset = None
    for dataset_name in dataset:
        if dataset_name == 'raddet':
            new_dataset = datasets.RADDet_Dataset(aug_params, split='test')
        val_dataset = new_dataset if val_dataset is None else val_dataset + new_dataset

    epe_list = []
    print('begin eval !')
    for val_id in tqdm(range(0, len(val_dataset), 20)):
        try:
            radar_cube_gt, radar_point, segment_mask = val_dataset[val_id]

            radar_point = radar_point.cuda().unsqueeze(0)
            segment_mask = segment_mask.cuda().unsqueeze(0)

            with autocast(enabled=mixed_prec):
                radar_cube = model(radar_point, segment_mask)

            radar_cube = radar_cube.squeeze(0).squeeze(0).cpu()

            assert radar_cube.shape == radar_cube_gt.shape, (radar_cube.shape, radar_cube_gt.shape)

            error_cube = (radar_cube - radar_cube_gt).abs()

            epe = error_cube.mean().item()

            # logging.info(f"{dataset} Iter {val_id+1} out of {len(val_dataset)}. epe {round(epe,4)}")
            
            epe_list.append(epe)
        except:
            continue

    epe_list = np.array(epe_list)

    print(f"Validation {dataset}: epe {round(epe_list.mean(),4)}")

    return {f'{dataset}-epe': round(epe_list.mean(),4), }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./checkpoints_icfar/icfar_raddet_bs3_hourglassv2/icfar-net.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='raddet', choices=['raddet', 'cruw', 'carada'])
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')

    # ICFARNet Settings    
    parser.add_argument('--hidden_dims', type=int, default=32, help="hidden_dims.")
    parser.add_argument('--output_dims', type=int, default=1, help="output_dims.")

    args = parser.parse_args()

    model = torch.nn.DataParallel(ICFARNet(args), device_ids=[0])

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info("Done loading checkpoint")

    model.cuda()
    model.eval()

    print(f"The model has {format(count_parameters(model)/1e6, '.2f')}M learnable parameters.")

    validate(model, mixed_prec=args.mixed_precision)
