#!/usr/bin/env python2.7

import os

def is_organ(label, organ_ID):
    """
    returning the binary label map by the organ ID (especially useful under overlapping cases).
    :param label:  the label matrix
    :param organ_ID:  the organ ID
    :return: binary label map
    """
    return label == organ_ID

def in_training_set(total_samples, i, folds, current_fold):
    '''
    determining if a sample belongs to the training set by the fold number
    :param total_samples: the total number of samples
    :param i: sample ID, an integer in [0, total_samples - 1]
    :param folds: the total number of folds
    :param current_fold: the current fold ID, an integer in [0, folds - 1]
    :return: True (False)
    '''
    fold_remainder = folds - total_samples % folds
    fold_size = (total_samples - total_samples % folds) / folds
    start_index = fold_size * current_fold + max(0, current_fold - fold_remainder)
    end_index = fold_size * (current_fold + 1) + max(0, current_fold + 1 - fold_remainder)
    return not (i >= start_index and i < end_index)

def training_set_filename(slice_path, current_fold):
    '''
    returning the filename of the training set according to the current fold ID
    :param current_fold: the current fold ID, an integer in [0, folds - 1]
    :return: path
    '''
    return os.path.join(slice_path, 'training_' + 'FD' + str(current_fold) + '.txt')

def testing_set_filename(slice_path, current_fold):
    '''
    returning the filename of the testing set according to the current fold ID
    :param current_fold: the current fold ID, an integer in [0, folds - 1]
    :return: path
    '''
    return os.path.join(slice_path, 'testing_' + 'FD' + str(current_fold) + '.txt')

def log_filename(snapshots_directory):
    '''
    returning the filename of the log file
    :param snapshots_directory:
    :return: path
    '''
    count = 0
    while True:
        count += 1
        if count == 1:
            log_file_ = os.path.join(snapshots_directory, 'log.txt')
        else:
            log_file_ = os.path.join(snapshots_directory, 'log' + str(count) + '.txt')
        if not (os.path.isfile(log_file_)):
            return log_file_
        else:
            raise ValueError('log file: {} already exists'.format(log_file_))

def valid_loss(log_file, iterations):
    '''
    determing if the loss values are reasonable (otherwise re-training is required)
    :param log_file: log filename
    :param iterations: current iterations
    :return: True or False
    '''
    FRACTION = 0.02
    loss_avg = 0.0
    loss_min = 1.0
    count = 0
    text = open(log_file, 'r').read().splitlines()
    for l in range(int(len(text) - iterations / 5 * FRACTION - 10), len(text)):
        index1 = text[l].find('Iteration')
        index2 = text[l].find('(')
        index3 = text[l].find('loss = ')
        if index1 > 0 and index2 > index1 and index3 > index2:
            iteration = int(text[l][index1 + 10: index2 - 1])
            loss = float(text[l][index3 + 7:])
            if iteration >= iterations * (1 - FRACTION):
                loss_avg += loss
                loss_min = min(loss_min, loss)
                count += 1
    if count > 0:
        loss_avg /= count
    else:
        loss_avg = loss
        loss_min = loss
    return loss_avg < 0.4 and loss_min < 0.35

def volumne_filename_testing(result_directory, t, i):
    '''
    returning the volumne filename as in the fusion stage
    :param result_directory:
    :param t:
    :param i:
    :return:
    '''
    return os.path.join(result_directory, str(t) + '_' + str(i + 1) + '.npz')

def DSC_computation(label, pred):
    '''
    computing the DSC together with other values based on the label and prediction volumes
    :param lable:
    :param pred:
    :return: dice
    '''
    # return label == pred

