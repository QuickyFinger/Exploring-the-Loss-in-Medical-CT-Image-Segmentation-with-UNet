#!/usr/bin/env python2.7

import torch
from torch import einsum
from scipy.spatial.distance import directed_hausdorff

def probs2one_hot(probs):
    '''
    :param probs:
    :return:
    '''
    return (probs >= 0.5).type(torch.int32)

def intersection(a, b):
    '''
    :param a:
    :param b:
    :return:
    '''
    return a & b

def dice_coef(probs, target, smooth=1e-8):
    '''
    :param probs:
    :param target:
    :return:
    '''
    inter_size = einsum('bcwh->b', (intersection(probs, target), )).type(torch.float32)
    sum_sizes = (einsum('bcwh->b', (target, )) + einsum('bcwh->b', (probs, ))).type(torch.float32)

    dices = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices

def hausdorff(probs, target):
    '''
    :param probs:
    :param target:
    :return:
    '''

    n_pred = probs.cpu().numpy()
    n_target = target.cpu().numpy()

    B, _, _, _ = probs.shape
    res = torch.zeros((B, ), dtype=torch.float32, device=probs.device)
    for b in range(B):
        res[b] = max(directed_hausdorff(n_pred[b, 0], n_target[b, 0])[0], directed_hausdorff(n_target[b, 0], n_pred[b, 0])[0])

    return res