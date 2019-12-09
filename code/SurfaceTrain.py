import torch
import os
import argparse
import math
import datetime
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from slice_data import *
from utils import training_set_filename, testing_set_filename, is_organ
from SurfaceUtils import label2dist
from SurfaceLoss import DistHausdorff, DiceLoss, GeneralizedDice
from SurfaceUnet import UNet, weights_init
import torch.nn.functional as F

def get_loss(dataloader, model):
    '''
    compute loss
    :param dataloader: data loader
    :param model: training model
    :return: diceloss, disthausdorff
    '''
    diceloss = GeneralizedDice()
    disthausdorff = DistHausdorff()
    dice = 0
    hausdorff = 0
    for i, (X, Y, D) in enumerate(dataloader):
        X, Y, D = Variable(X).cuda(), Variable(Y).cuda(), Variable(D).cuda()
        output = model(X)
        probs = F.sigmoid(output)
        diceloss_ = diceloss(probs, Y).cpu().detach().numpy()
        disthausdorff_ = disthausdorff(probs, Y, D).cpu().detach().numpy()
        dice += diceloss_
        hausdorff += disthausdorff_
    dice = dice / len(dataloader)
    hausdorff = hausdorff / len(dataloader)
    return dice, hausdorff

class CTDataset(Dataset):
    '''
    CT dataset interface
    '''
    def __init__(self, current_fold=0, low_range=-100, high_range=240, organ_ID=1, slice_threshold=0.98, slice_path=None, train=True):
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
        self.positive = 0
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

        # distance transform
        dist_maps = label2dist(label)

        # expand dims
        image = image[np.newaxis, ...]
        label = label[np.newaxis, ...]
        dist_maps = dist_maps[np.newaxis, ...]

        # numpy2tensor
        image = torch.Tensor(image)
        label = torch.Tensor(label)
        dist_maps = torch.Tensor(dist_maps)

        return image, label, dist_maps

    def __len__(self):
        '''
        :return: dataset length
        '''
        return len(self.image_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--depth', type=int, default=5, help='UNet depth')
    parser.add_argument('--start_filts', type=int, default=64, help='unet start filters')
    parser.add_argument('--merge_mode', type=str, default='concat', help='merge mode, concat or addition')
    parser.add_argument('--lr', type=float, default=5e-4, help='optimizer learning rate')
    # parser.add_argument('--log_interval', type=int, default=5200, help='less than len(train_dataloader)')
    parser.add_argument('--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='number of samples in a batch')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_path', type=str, default='/home/redhand/DC_Comptition/CT-pancreas-segmentation/metric-loss/Hausdorff/data2npy')
    parser.add_argument('--slice_threshold', type=float, default=0.98)
    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--current_fold', type=int, default=0, help='the current fold, [0, 3], default 0')
    parser.add_argument('--low_range', type=int, default=-100, help='low boundary for normalization, default -100')
    parser.add_argument('--high_range', type=int, default=240, help='high boundary for normalization, default 200')
    parser.add_argument('--organ_ID', type=int, default=1, help='organ index, default 1')
    parser.add_argument('--dice_weight', type=float, default=0.9, help='dice loss coefficient')
    parser.add_argument('--hausdorff_weight', type=float, default=0.1, help='huasdorff loss coefficient')
    parser.add_argument('--csv', type=str, default='./logs')
    config = parser.parse_args()

    slice_path = os.path.join(config.data_path, 'slices')
    train_dataloader = DataLoader(CTDataset(current_fold=config.current_fold,
                                            low_range=config.low_range,
                                            high_range=config.high_range,
                                            organ_ID=config.organ_ID,
                                            slice_threshold=config.slice_threshold,
                                            slice_path=slice_path,
                                            train=True),
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  pin_memory=torch.cuda.is_available(),
                                  num_workers=config.num_workers)
    val_dataloader = DataLoader(CTDataset(current_fold=config.current_fold,
                                          low_range=config.low_range,
                                          high_range=config.high_range,
                                          organ_ID=config.organ_ID,
                                          slice_threshold=config.slice_threshold,
                                          slice_path=slice_path,
                                          train=False),
                                batch_size=config.batch_size,
                                shuffle=False,
                                pin_memory=torch.cuda.is_available(),
                                num_workers=config.num_workers)
    model = UNet(nin=1, nout=1)
    model.apply(weights_init)
    model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.99), amsgrad=False)
    model.train()
    # initialize log parameters
    iter = 0
    iteration = []
    train_losses = []
    print('check model log directory ...')
    if not os.path.exists(config.csv):
        os.makedirs(config.csv)
    print('check model save directory ...')
    date = datetime.datetime.now().strftime('%Y-%m-%d')
    model_path = os.path.join(config.model_path, date)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # diceloss = DiceLoss()
    disthausdorff = DistHausdorff()
    GDE = GeneralizedDice()
    train_loss = np.zeros((config.epochs, len(train_dataloader)))
    val_loss = np.zeros((config.epochs, ))
    val_diceloss = np.zeros((config.epochs, ))
    val_hausdorff = np.zeros((config.epochs, ))
    for epoch in range(config.epochs):
        for i, (X, Y, D) in enumerate(train_dataloader):
            # forward pass
            X = Variable(X).cuda()  # [N, 1, H, W]
            Y = Variable(Y).cuda()  # [N, 1, H, W] with class indices (0, 1)
            D = Variable(D).cuda()  # [N, 1, H, W]
            logit = model(X)   # [N, 1, H, W]
            output = F.sigmoid(logit)
            loss = config.dice_weight * GDE(output, Y) + config.hausdorff_weight * disthausdorff(output, Y, D)

            # backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()

            # validaton
            train_loss[epoch][i] = loss.data
            if (i + 1) % len(train_dataloader) == 0:
            # if True:
                # iter += config.log_interval * config.batch_size
                # iteration.append(iter)
                train_losses_ = train_loss[epoch].mean()
                model.eval()
                print('getting validation loss ...')
                val_diceloss_, val_disthausdorff_ = get_loss(val_dataloader, model)
                val_loss_ = config.dice_weight * val_diceloss_ + config.hausdorff_weight * val_disthausdorff_
                model.train()                # if val_loss < min_loss:
                date = datetime.datetime.now().strftime('%H-%M-%S')
                best_model_file_path = os.path.join(model_path, 'unet-loss-' + '{:.4f}'.format(val_loss_) + \
                                                    '-epoch{:0>4}'.format(epoch) + '-' + date + '.model')
                torch.save(model.state_dict(), best_model_file_path)
                # min_loss = val_loss
                print('epoch: {}, iteration: {}, train loss: {:.4f}, validation dice loss: {:.4f}, '
                      'validation dist huasdorff loss: {:.4f}, weighted loss: {:.4f}'.format(epoch+1, \
                        i+1, train_losses_, val_diceloss_, val_disthausdorff_, val_loss_))
                val_loss[epoch] = val_loss_
                val_diceloss[epoch] = val_diceloss_
                val_hausdorff[epoch] = val_disthausdorff_
        df = pd.DataFrame({'train_loss': train_loss.mean(axis=1),
                           'val_loss': val_loss,
                           'val_diceloss': val_diceloss,
                           'val_hausdorff': val_hausdorff})
        date = datetime.datetime.now().strftime('%Y-%m-%d')
        df.to_csv(os.path.join(config.csv, date + '-loss.csv'), float_format="%.4f", index_label="epoch")
    print('training samples: {}'.format(len(train_dataloader)))


