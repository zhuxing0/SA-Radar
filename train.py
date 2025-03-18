import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from core.ICFARNet import ICFARNet
from evaluate import validate, count_parameters
import core.datasets as datasets
import torch.nn.functional as F
import pdb

try:
    from torch.cuda.amp import GradScaler
except:
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def sequence_loss(radar_cube_gt, radar_cube, segment_mask, args):
    """ Loss function
        radar_cube_gt: (B, R, D, A) , float()
        radar_cube: (B, 1, R, D, A)
        segment_mask: (B, 1, R, D, A)
    """

    radar_cube = radar_cube.squeeze(1)

    loss = 0.0

    smooth_l1_loss = 1.0 * F.smooth_l1_loss(radar_cube, radar_cube_gt, size_average=True)

    error_cube = (radar_cube - radar_cube_gt).abs()
    l1_loss = error_cube.mean()

    loss += smooth_l1_loss
    loss += l1_loss

    if args.segment_mask_loss:
        segment_mask = segment_mask.squeeze(1)
        segment_mask = segment_mask > 0.5
        if torch.sum(segment_mask) == 0:
            smooth_l1_loss_in_bbox = torch.zeros_like(loss)
            l1_loss_in_bbox = torch.zeros_like(loss)
        else:
            radar_cube_in_bbox = radar_cube[segment_mask]
            radar_cube_gt_in_bbox = radar_cube_gt[segment_mask]
            smooth_l1_loss_in_bbox = 1.0 * F.smooth_l1_loss(radar_cube_in_bbox, radar_cube_gt_in_bbox, size_average=True)
            error_cube_in_bbox = (radar_cube_in_bbox - radar_cube_gt_in_bbox).abs()
            l1_loss_in_bbox = error_cube_in_bbox.mean()
        if args.sml1_loss:
            loss += smooth_l1_loss_in_bbox
        if args.l1_loss:
            loss += l1_loss_in_bbox
        metrics = {
            'loss': loss.item(),
            'smooth_l1_loss': smooth_l1_loss.item(),
            'l1_loss': l1_loss.item(),
            'smooth_l1_loss_in_bbox': smooth_l1_loss_in_bbox.item(),
            'l1_loss_in_bbox': l1_loss_in_bbox.item()
        }
    else:
        metrics = {
            'loss': loss.item(),
            'smooth_l1_loss': smooth_l1_loss.item(),
            'l1_loss': l1_loss.item()
        }
    if torch.isnan(loss):
        pdb.set_trace()

    return loss, metrics

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
    
    return optimizer, scheduler 

class Logger:
    def __init__(self, model, scheduler, SUM_FREQ=99):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.SUM_FREQ = SUM_FREQ
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=args.logdir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/self.SUM_FREQ for k in self.running_loss.keys()]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=args.logdir)

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/self.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.SUM_FREQ == self.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}
        
        self.total_steps += 1

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=args.logdir)

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = ICFARNet(args) # nn.DataParallel()
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = datasets.fetch_dataloader(args)
    args.num_steps = args.epochs * len(train_loader)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        model.load_state_dict(checkpoint, strict=True)
        logging.info("Done loading checkpoint")
    model.cuda()
    model.train()

    validation_frequency = 200 # len(train_loader)

    logger = Logger(model, scheduler, 99)

    scaler = GradScaler(enabled=args.mixed_precision)

    should_keep_training = True
    global_batch_num = 0
    while should_keep_training:

        for i_batch, data_blob in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            
            if args.attribute:
                radar_cube_gt, radar_point, segment_mask, sigma, g, Rs, lambda_ = [x.cuda() for x in data_blob]
            else:
                radar_cube_gt, radar_point, segment_mask = [x.cuda() for x in data_blob] # # (B, R, D, A), (B, 1, R, D, A), (B, 1, R, D, A)

            assert model.training
            if args.attribute:
                radar_cube = model(radar_point, sigma, g, Rs, lambda_)
            else:
                radar_cube = model(radar_point) # (B, output_dims, R, D, A)
            assert model.training

            loss, metrics = sequence_loss(radar_cube_gt, radar_cube, segment_mask, args)

            logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            logger.writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            global_batch_num += 1
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            logger.push(metrics)

            if total_steps % validation_frequency == validation_frequency - 1: # or total_steps == 5

                if total_steps % (10*(args.epochs/10)*validation_frequency) == (10*(args.epochs/10)*validation_frequency) - 1:
                    save_path = Path(args.logdir + '/%d_%s.pth' % (total_steps + 1, args.name))
                    logging.info(f"Saving file {save_path.absolute()}")
                    torch.save(model.state_dict(), save_path)

                results = validate(model, args.train_datasets)
                logger.write_dict(results)

                model.train()

            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    print("FINISHED TRAINING")
    logger.close()
    PATH = args.logdir + '/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='icfar-net', help="name your experiment")
    parser.add_argument('--restore_ckpt', default=None, help="load the weights from a specific checkpoint")
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--logdir', default='./checkpoints/icfar', help='the directory to save logs and checkpoints')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=3, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['raddet'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.0002, help="max learning rate.")
    parser.add_argument('--epochs', type=int, default=50, help="length of training schedule.")
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # ICFARNet Settings
    parser.add_argument('--hidden_dims', type=int, default=32, help="hidden_dims.")
    parser.add_argument('--output_dims', type=int, default=1, help="output_dims.")
    parser.add_argument('--attribute', action='store_true', help="attribute embedding or not")

    # loss Settings
    parser.add_argument('--segment_mask_loss', action='store_true', help='use segment_mask_loss')
    parser.add_argument('--l1_loss', action='store_true', help="")
    parser.add_argument('--sml1_loss', action='store_true', help="")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=[0, 1.4], help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[-0.2, 0.4], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    args = parser.parse_args()

    torch.manual_seed(666)
    np.random.seed(666)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path(args.logdir).mkdir(exist_ok=True, parents=True)

    train(args)
