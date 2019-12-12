import numpy as np
import os
import time
import argparse
from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/redhand/DC_Comptition/CT-pancreas-segmentation/metric-loss/Hausdorff/data2npy')
    parser.add_argument('--organ_number', type=int, default=1)
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--low_range', type=int, default=-100)
    parser.add_argument('--high_range', type=int, default=240)
    config = parser.parse_args()
    image_list = []
    label_list = []
    image_filename = []
    label_filename = []
    keyword = ''
    image_path = os.path.join(config.data_path, 'images')
    label_path = os.path.join(config.data_path, 'labels')

    for directory, _, file_ in os.walk(image_path):
        for filename in sorted(file_):
            if keyword in filename:
                image_list.append(os.path.join(directory, filename))
                image_filename.append(os.path.splitext(filename)[0])
    for directory, _, file_ in os.walk(label_path):
        for filename in sorted(file_):
            if keyword in filename:
                label_list.append(os.path.join(directory, filename))
                label_filename.append(os.path.splitext(filename)[0])

    if len(image_list) != len(label_list):
        exit('Error: the number of labels and the number of images are not equal! {}, {}'.format(len(image_list), len(label_list)))

    total_samples = len(image_list)
    # slice path
    slice_path = os.path.join(config.data_path, 'slices')
    if not os.path.exists(slice_path):
        os.makedirs(slice_path)
    slice_file = os.path.join(slice_path, 'training_slices.txt')
    image_path_ = os.path.join(config.data_path, 'images_slices')
    if not os.path.exists(image_path_):
        os.makedirs(image_path_)
    label_path_ = os.path.join(config.data_path, 'label_slices')
    if not os.path.exists(label_path_):
        os.makedirs(label_path_)
    output = open(slice_file, 'w')
    output.close()

    print 'Initialization starts'
    for i in range(total_samples):
        start_time = time.time()
        print 'Processing ' + str(i + 1) + ' out of ' + str(len(image_list)) + ' files.'
        image = np.load(image_list[i])
        label = np.load(label_list[i])
        print ' 3D volume is loaded: ' + str(time.time() -  start_time) + ' second(s) elapsed.'
        slice_number = label.shape[2]
        image_directory_ = os.path.join(image_path_, image_filename[i])
        if not os.path.exists(image_directory_):
            os.makedirs(image_directory_)
        label_directory_ = os.path.join(label_path_, label_filename[i])
        if not os.path.exists(label_directory_):
            os.makedirs(label_directory_)
        print '    Slicing data: ' + str(time.time() -  start_time) + ' second(s) elapsed.'
        sum_ = np.zeros((slice_number, config.organ_number + 1), dtype = np.int)
        minA = np.zeros((slice_number, config.organ_number + 1), dtype = np.int)
        maxA = np.zeros((slice_number, config.organ_number + 1), dtype = np.int)
        minB = np.zeros((slice_number, config.organ_number + 1), dtype = np.int)
        maxB = np.zeros((slice_number, config.organ_number + 1), dtype = np.int)
        average = np.zeros((slice_number), dtype = np.float)
        for j in range(0, slice_number):
            image_filename_ = os.path.join(\
                image_path_, image_filename[i], '{:0>4}'.format(j) + '.npy')
            label_filename_ = os.path.join(\
                label_path_, label_filename[i], '{:0>4}'.format(j) + '.npy')
            image_ = image[:, :, j]
            label_ = label[:, :, j]
            if not os.path.isfile(image_filename_) or not os.path.isfile(label_filename_):
                np.save(image_filename_, image_)
                np.save(label_filename_, label_)
            np.minimum(np.maximum(image_, config.low_range, image_), config.high_range, image_)
            average[j] = float(image_.sum()) / (image_.shape[0] * image_.shape[1])
            for o in range(1, config.organ_number + 1):
                sum_[j, o] = (is_organ(label_, o)).sum()
                arr = np.nonzero(is_organ(label_, o))
                minA[j, o] = 0 if not len(arr[0]) else min(arr[0])
                maxA[j, o] = 0 if not len(arr[0]) else max(arr[0])
                minB[j, o] = 0 if not len(arr[1]) else min(arr[1])
                maxB[j, o] = 0 if not len(arr[1]) else max(arr[1])
        print '   Writing training lists: ' + str(time.time() - start_time) + ' second(s) elapsed.'
        output = open(slice_file, 'a+')
        for j in range(0, slice_number):
            image_filename_ = os.path.join(\
                image_path_, image_filename[i], '{:0>4}'.format(j) + '.npy')
            label_filename_ = os.path.join(\
                label_path_, label_filename[i], '{:0>4}'.format(j) + '.npy')
            output.write(str(i) + ' ' + str(j))
            output.write(' ' + image_filename_ + ' ' + label_filename_)
            output.write(' ' + str(average[j]))
            for o in range(1, config.organ_number + 1):
                output.write(' ' + str(sum_[j, o]) + ' ' + str(minA[j, o]) + \
                             ' ' + str(maxA[j, o]) + ' ' + str(minB[j, o]) + ' ' + str(maxB[j, o]))
            output.write('\n')
        output.close()
        print 'Processed ' + str(i + 1) + ' out of ' + str(len(image_list)) + ' files: ' + \
            str(time.time() - start_time) + 'second(s) elapsed.'

    print 'Writing training image list.'
    for f in range(config.folds):
        list_training_ = training_set_filename(slice_path, f)
        output = open(list_training_, 'w')
        for i in range(total_samples):
            if in_training_set(total_samples, i, config.folds, f):
                output.write(str(i) + ' ' + image_list[i] + ' ' + label_list[i] + '\n')
        output.close()

    print 'Writing testing image list.'
    for f in range(config.folds):
        list_testing = testing_set_filename(slice_path, f)
        output = open(list_testing, 'w')
        for i in range(total_samples):
            if not in_training_set(total_samples, i, config.folds, f):
                output.write(str(i) + ' ' + image_list[i] + ' ' + label_list[i] + '\n')
        output.close()
    print 'Initialization is done.'
