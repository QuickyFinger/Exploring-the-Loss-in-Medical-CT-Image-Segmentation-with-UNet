import torch
import argparse
import os
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from SurfaceUnet import UNet
from BoundaryTrain import CTDataset
import torch.nn.functional as F

def get_output(dataloader, model, config):
    '''
    compute loss
    :param dataloader: data loader
    :param model: training model
    :return: loss
    '''
    for i, (X, Y) in enumerate(dataloader):
        X, Y = Variable(X).cuda(), Variable(Y).cuda()
        output = model(X)
        output = F.sigmoid(output)
        np.save(os.path.join(config.output_path, 'predict_' + '{:0>4}'.format(i) + '.npy'), output.cpu().detach().numpy())
        np.save(os.path.join(config.output_path, 'label_' + '{:0>4}'.format(i) + '.npy'), Y.cpu().detach().numpy())
        np.save(os.path.join(config.output_path, 'input_' + '{:0>4}'.format(i) + '.npy'), X.cpu().detach().numpy())
        print('completing saving {}'.format(i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--best_model_file_path', type=str, default='./boundary-models/2019-12-11/unet-loss-0.7502-epoch0032-10-59-32.model')
    parser.add_argument('--current_fold', type=int, default=0, help='the current fold, [0, 3], default 0')
    parser.add_argument('--low_range', type=int, default=-100, help='low boundary for normalization, default -100')
    parser.add_argument('--high_range', type=int, default=240, help='high boundary for normalization, default 200')
    parser.add_argument('--slice_path', type=str, default='./slices')
    parser.add_argument('--organ_ID', type=int, default=1, help='organ index, default 1')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--slice_threshold', type=float, default=0.98)
    parser.add_argument('--batch_size', type=int, default=1, help='number of samples in a batch')
    parser.add_argument('--output_path', type=str, default='./boundary-output')
    config = parser.parse_args()
    if not os.path.exists(config.output_path):
        print('make output dirs ...')
        os.makedirs(config.output_path)
    slice_path = os.path.join(config.data_path, 'slices')
    model = UNet(nin=1, nout=1)
    model.cuda()
    model.load_state_dict(torch.load(config.best_model_file_path))
    model.eval()
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
    print('getting val loss ...')
    get_output(val_dataloader, model, config)
