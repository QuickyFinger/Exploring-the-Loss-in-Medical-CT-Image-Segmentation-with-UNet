#!/usr/bin/env python2.7

import torch
import os
import argparse
import math
import datetime
import numpy as np
import pandas as pd
from functools import reduce
from operator import add
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from SliceData import *
from utils import training_set_filename, testing_set_filename, is_organ
from SurfaceLoss import DiceLoss
from BoundaryLoss import BoundaryLoss
from BoundaryUtils import dice_coef, hausdorff, probs2one_hot
from SurfaceUnet import UNet, weights_init
import torch.nn.functional as F

class CTDataset(Dataset):
    '''
    CT dataset interface
    '''
    def __init__(self, current_fold=0, low_range=-100, high_range=240, organ_ID=1, slice_threshold=0.10, slice_path=None, train=True):
        '''
        :param current_fold: int, current fold, [0-3], default 0
        :param low_range: int, low boundary for normalization, default -100
        :param high_range: int, high boundary for normalization, default 240
        :param organ_ID: int, current organ index, default 1
        :param slice_path: str, slice path, default None
        :param train: bool, training or testing, default true
        '''
        # initialize
        self.current_fold = current_fold
        self.low_range = low_range
        self.high_range = high_range
        self.organ_ID = organ_ID
        self.slice_path = slice_path
        self.slice_threshold = slice_threshold
        self.train = train
        # load fold data
        if self.train:
            self.image_list = open(training_set_filename(self.slice_path, self.current_fold), 'r').read().splitlines()
        else:
            self.image_list = open(testing_set_filename(self.slice_path, self.current_fold), 'r').read().splitlines()
        self.image_set = np.zeros((len(self.image_list)), dtype=np.int)
        for i in range(len(self.image_list)):
            s = self.image_list[i].split(' ')
            self.image_set[i] = s[0]
        if self.train:
            print('training image set: {}'.format(self.image_set))
        else:
            print('test image set: {}'.format(self.image_set))
        self.slice_list = open(os.path.join(self.slice_path, 'training_slices.txt'), 'r').read().splitlines()
        self.slices = len(self.slice_list)
        self.image_ID = np.zeros((self.slices), dtype=np.int)
        image_filename_ = ['' for l in range(self.slices)]
        label_filename_ = ['' for l in range(self.slices)]
        self.pixels = np.zeros((self.slices), dtype=np.int)
        for l in range(self.slices):
            s = self.slice_list[l].split(' ')
            self.image_ID[l] = s[0]
            image_filename_[l] = s[2]
            label_filename_[l] = s[3]
            self.pixels[l] = s[self.organ_ID * 5]
        if len(image_filename_) != len(label_filename_):
            raise ValueError('image length is not euqal with label length! image length {}, \
                                label length {}'.format(len(image_filename_), len(label_filename_)))
        # select data
        if self.train:
            if self.slice_threshold <= 1:
                self.pixels_index = sorted(range(self.slices), key=lambda l: self.pixels[l])
                self.last_index = int(math.floor((self.pixels > 0).sum() * self.slice_threshold))
                self.min_pixels = self.pixels[self.pixels_index[-self.last_index]]
            else:
                self.min_pixels = self.slice_threshold
            self.active_index = [l for l, p in enumerate(self.pixels) if p >= self.min_pixels]
            self.image_filename = []
            self.label_filename = []
            for i in self.active_index:
                if self.image_ID[i] in self.image_set:
                    self.image_filename.append(image_filename_[i])
                    self.label_filename.append(label_filename_[i])
        else:
            self.image_filename = []
            self.label_filename = []
            for i in range(self.slices):
                if self.image_ID[i] in self.image_set:
                    self.image_filename.append(image_filename_[i])
                    self.label_filename.append(label_filename_[i])

    def __getitem__(self, index):
        '''
        pytorch: [batch, channel, height, width]
        :param index: image filename index, label filename index
        :return: (image, label)
        '''

        # seed = np.random.randint(2019)
        image = np.load(self.image_filename[index])
        label = np.load(self.label_filename[index])

        # clip image
        np.minimum(np.maximum(image, self.low_range, image), self.high_range, image)

        # normalize image
        image -= self.low_range
        image = image*1.0 / (self.high_range - self.low_range)
        image = image.astype(np.float32)
        label = is_organ(label, self.organ_ID).astype(np.float32)

        # expand dims
        image = image[np.newaxis, ...]
        label = label[np.newaxis, ...]

        # numpy2tensor
        image = torch.Tensor(image)
        label = torch.Tensor(label)

        return image, label

    def __len__(self):
        '''
        :return: dataset length
        '''
        return len(self.image_filename)

def do_epoch(model, dataloader, loss_fns, loss_weights, optimizer=None, compute_hausdorff=False):

    if optimizer:
        model.train()
    else:
        model.eval()

    total_iteration, total_images = len(dataloader), len(dataloader.dataset)
    all_dices = torch.zeros((total_images, ), dtype=torch.float32, device=torch.device("cuda"))
    loss_log = torch.zeros((total_iteration, ), dtype=torch.float32, device=torch.device("cuda"))
    hausdorff_log = torch.zeros((total_images, ), dtype=torch.float32, device=torch.device("cuda"))
    done = 0

    # batch training
    for i, (X, Y) in enumerate(dataloader):
        if optimizer:
            optimizer.zero_grad() # Reset Gradients.
        # Forward
        X = Variable(X).cuda()  # [N, 1, H, W]
        Y = Variable(Y).cuda()  # [N, 1, H, W] with class indices (0, 1)
        logit = model(X)  # [N, 1, H, W]
        output = F.sigmoid(logit)
        predicted_mask = probs2one_hot(output.detach())
        ziped = zip(loss_fns, loss_weights)
        losses = [w * loss_fn(output, Y) for loss_fn, w in ziped]
        loss = reduce(add, losses)
        # loss = args.dice_weight * Dice(output, Y) + args.boundary_weight * Boundary(output, Y)
        # Backward
        if optimizer:
            loss.backward()
            optimizer.step()
        loss_log[i] = loss.detach()
        dices = dice_coef(predicted_mask, Y.detach().type(torch.int32))
        B = len(X)
        all_dices[done: done+B] = dices
        if compute_hausdorff:
            hausdorff_res = hausdorff(predicted_mask.detach(), Y.detach().type(torch.int32))
            hausdorff_log[done: done+B] = hausdorff_res
        done = done + B

    return all_dices, loss_log, hausdorff_log

def run(args):
    '''
    :param args:
    :return:
    '''
    # slices, image filename and label filename
    slice_path = os.path.join(args.data_path, 'slices')

    # initialize dataloader
    train_dataloader = DataLoader(CTDataset(current_fold=args.current_fold,
                                            low_range=args.low_range,
                                            high_range=args.high_range,
                                            organ_ID=args.organ_ID,
                                            slice_threshold=args.slice_threshold,
                                            slice_path=slice_path,
                                            train=True),
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=torch.cuda.is_available(),
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(CTDataset(current_fold=args.current_fold,
                                          low_range=args.low_range,
                                          high_range=args.high_range,
                                          organ_ID=args.organ_ID,
                                          slice_threshold=args.slice_threshold,
                                          slice_path=slice_path,
                                          train=False),
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=torch.cuda.is_available(),
                                num_workers=args.num_workers)

    # initiallize model
    model = UNet(nin=args.nin, nout=args.nout)
    model.apply(weights_init)
    model.cuda()

    # initialize optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), amsgrad=False)
    lr = args.lr

    # initialize model and log folder
    print('check model log directory ...')
    if not os.path.exists(args.csv):
        os.makedirs(args.csv)
    print('check model save directory ...')
    date = datetime.datetime.now().strftime('%Y-%m-%d')
    model_path = os.path.join(args.model_path, date)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # initialize loss
    Boundary = BoundaryLoss()
    Dice = DiceLoss()

    print('training samples (batch size, 2): {}'.format(len(train_dataloader)))

    n_tra = len(train_dataloader.dataset)  # Number of images in dataset
    l_tra = len(train_dataloader)  # Number of iteration per epoch: different if batch_size > 1
    n_val = len(val_dataloader.dataset)
    l_val = len(val_dataloader)

    best_dice = torch.zeros(1).to(torch.device("cuda")).type(torch.float32)
    best_epoch = 0

    metrics = {"val_dice": torch.zeros((args.epochs, n_val), device=torch.device("cuda")).type(torch.float32),
               "val_loss": torch.zeros((args.epochs, l_val), device=torch.device("cuda")).type(torch.float32),
               "val_hausdorff": torch.zeros((args.epochs, n_val), device=torch.device('cuda')).type(torch.float32),
               "tra_dice": torch.zeros((args.epochs, n_tra), device=torch.device("cuda")).type(torch.float32),
               "tra_loss": torch.zeros((args.epochs, l_tra), device=torch.device("cuda")).type(torch.float32),}

    loss_fns = [Dice, Boundary]
    loss_weights = [args.dice_weight, args.boundary_weight]

    for i in range(args.epochs):
        # Do training and validation loss
        tra_loss, tra_dice, _ = do_epoch(model, train_dataloader, loss_fns, loss_weights, optimizer=optim, compute_hausdorff=False)
        with torch.no_grad():
            val_loss, val_dice, val_hausdorff = do_epoch(model, val_dataloader, loss_fns, loss_weights, compute_hausdorff=True)
        # sort and save metrics
        for k in metrics:
            metrics[k][i] = eval(k)
        df = pd.DataFrame({'tra_loss': metrics['tra_loss'].mean(dim=1).cpu().numpy(),
                           'val_loss': metrics['val_loss'].mean(dim=1).cpu().numpy(),
                           'tra_dice': metrics['tra_dice'].mean(dim=1).cpu().numpy(),
                           'val_dice': metrics['val_dice'].mean(dim=1).cpu().numpy(),
                           'val_hausdorff': metrics['val_hausdorff'].mean(dim=1).cpu().numpy()})
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        df.to_csv(os.path.join(args.csv, date + '-loss.csv'), float_format="%.4f", index_label="epoch")
        current_dice = val_dice.mean()
        if current_dice > best_dice:
            best_dice = current_dice
            best_epoch = i + 1
            best_model_file_path = os.path.join(model_path, 'Unet-with-Boundary-Loss-Dice-{:.4f}-Epoch-{:0>4}'.\
                                                format(current_dice.cpu().numpy(), best_epoch))
            torch.save(model.state_dict(), best_model_file_path)

        if args.lr_schedule and (i % (best_epoch + 20) == 0):
            lr = lr * 0.5
            for param_group in optim.param_groups:
                param_group['lr'] = lr
                print('new learning rate: {}'.format(lr))

        print('epoch: {:0>4}, tra_loss: {:.4f}, val_loss: {:.4f}, tra_dice: {:.4f}, val_dice: {:.4f}, val_hausdorff: {:.4f}'.\
              format(i+1, tra_loss.mean().cpu().numpy(), val_loss.mean().cpu().numpy(),tra_dice.mean().cpu().numpy(),\
                     val_dice.mean().cpu().numpy(), val_hausdorff.mean().cpu().numpy()))

def get_args():
    '''
    :return:
    '''
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--lr', type=float, default=5e-4, help='optimizer learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='number of samples in a batch')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--slice_threshold', type=float, default=0.75)
    parser.add_argument('--model_path', type=str, default='./boundary-models')
    parser.add_argument('--current_fold', type=int, default=0, help='the current fold, [0, 3], default 0')
    parser.add_argument('--low_range', type=int, default=-100, help='low boundary for normalization, default -100')
    parser.add_argument('--high_range', type=int, default=240, help='high boundary for normalization, default 200')
    parser.add_argument('--organ_ID', type=int, default=1, help='organ index, default 1')
    parser.add_argument('--dice_weight', type=float, default=0.9, help='dice loss coefficient')
    parser.add_argument('--boundary_weight', type=float, default=0.1, help='huasdorff loss coefficient')
    parser.add_argument('--csv', type=str, default='./boundary-logs')
    parser.add_argument('--nin', type=int, default=1, help='input channel')
    parser.add_argument('--nout', type=int, default=1, help='output channel')
    parser.add_argument('--lr_schedule', type=bool, default=True, help='learning rate scheduler')
    args = parser.parse_args()
    print(args)
    return args

if __name__ == '__main__':
    run(get_args())
