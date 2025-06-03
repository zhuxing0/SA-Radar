import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
from core.ICFARNet import ICFARNet, autocast
import core.datasets as datasets

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def validate(model, dataset_all=['raddet'], mixed_prec=False, attribute=False):
    """ Peform validation using the raddet dataset """
    model.eval()
    aug_params = {}

    dataset = []
    for dataset_name in dataset_all:
        if 'raddet' in dataset_name and 'raddet' not in dataset:
            dataset.append('raddet')
        if 'carrada' in dataset_name and 'carrada' not in dataset:
            dataset.append('carrada')
        if 'raddet' not in dataset_name and 'carrada' not in dataset_name:
            dataset.append(dataset_name)

    val_dataset = None
    for dataset_name in dataset:
        if dataset_name == 'raddet':
            new_dataset = datasets.RADDet_Dataset(aug_params, split='test', attribute=attribute, Attributes=datasets.compute_the_Attribute(sigma_r_bias=0.057, N_list_bias=0, num_log_a_bias=0))
        elif dataset_name == 'carrada':
            new_dataset = datasets.Carrada_Dataset(aug_params, split='test', attribute=attribute, Attributes=datasets.compute_the_Attribute(sigma_r_bias=0.2, N_list_bias=0, num_log_a_bias=0))
        else:
            continue
        val_dataset = new_dataset if val_dataset is None else val_dataset + new_dataset
    
    epe_list = []
    epe_obj_list = []
    print('begin eval !')
    for val_id in tqdm(range(0, len(val_dataset))): # , 20
        try:
            if attribute:
                radar_cube_gt, radar_point, segment_mask, sigma, g, Rs, lambda_ = val_dataset[val_id]
                sigma, g, Rs, lambda_ = sigma.cuda().unsqueeze(0), g.cuda().unsqueeze(0), Rs.cuda().unsqueeze(0), lambda_.cuda().unsqueeze(0)
            else:
                radar_cube_gt, radar_point, segment_mask = val_dataset[val_id]

            radar_point = radar_point.cuda().unsqueeze(0)
            segment_mask = segment_mask.cuda().unsqueeze(0)

            with autocast(enabled=mixed_prec):
                if attribute:
                    radar_cube = model(radar_point, sigma, g, Rs, lambda_)
                else:
                    radar_cube = model(radar_point)

            radar_cube = radar_cube.squeeze(0).squeeze(0).cpu()

            assert radar_cube.shape == radar_cube_gt.shape, (radar_cube.shape, radar_cube_gt.shape)

            error_cube = (radar_cube - radar_cube_gt).abs()

            epe = error_cube.mean().item()

            obj_mask = segment_mask.squeeze(0).squeeze(0).cpu() > 0.5
            epe_obj = error_cube[obj_mask].mean().item()

            # logging.info(f"{dataset} Iter {val_id+1} out of {len(val_dataset)}. epe {round(epe,4)}")
            epe_list.append(epe)
            if not np.isnan(epe_obj):
                epe_obj_list.append(epe_obj)
        except:
            continue

    epe_list = np.array(epe_list)
    epe_obj_list = np.array(epe_obj_list)

    print(f"Validation {dataset}: epe {round(epe_list.mean(),4)}, epe_obj {round(epe_obj_list.mean(),4)}")

    return {f'{dataset}-epe': round(epe_list.mean(),4), f'{dataset}-epe_obj': round(epe_obj_list.mean(),4)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./models/icfar-net.pth')
    parser.add_argument('--dataset', help="dataset for evaluation", default='raddet', choices=['raddet', 'carrada'])
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')

    # ICFARNet Settings    
    parser.add_argument('--hidden_dims', type=int, default=32, help="hidden_dims.")
    parser.add_argument('--output_dims', type=int, default=1, help="output_dims.")
    parser.add_argument('--attribute', action='store_true', help="attribute embedding or not")

    args = parser.parse_args()

    model = ICFARNet(args) # torch.nn.DataParallel(ICFARNet(args), device_ids=[0])

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

    validate(model, mixed_prec=args.mixed_precision, attribute=args.attribute)
