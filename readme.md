# Exploring the Loss Functions in Medical CT Image Segmentation with UNet
Code for our medical CT image segmentation project. Thanks for the following sharing: 
1. [An end-to-end coarse-to-fine framework for organ segmentation: OrganSegRSTN](https://github.com/198808xc/OrganSegRSTN)
2. [H-DenseUNet: Hybrid Densely Connected UNet for Liver and Tumor Segmentation from CT Volumes ](https://github.com/xmengli999/H-DenseUNet)
3. [Boundary Loss for Highly Unbalanced Segmentation](https://github.com/LIVIAETS/surface-loss)
## Requirements
None-exhaustive list:
* python 2.7
* Pytorch 0.4.0
* nibabel
* Scipy
* Numpy
* dicom

## Usage
### Data Preprocessing
Download dataset from [Pancreas-CT](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT). Then unzip file in the command line: 
```
unzip TCIA_pancreas_labels-02-05-2017.zip
```
Convert image data into .npy format:
```
python dicom2npy.py
```
Convert label data into .npy format:
```
python nii2npy.py
```
Slice data along the z-axis: 
```
python SliceData.py
```
image_slices, label_slices and slices folders will be generated:

| Folder/File                           | Description                                                   |
|:--------------------------------------|:--------------------------------------------------------------|
|**image_slices/**                      |storing image slices                                           |
|**label_slices/**                      |storing label slices                                           |
|**slices/**                            |storing training and test fold information                     |
|`training_slices.txt`                  |storing image filenames and label filenames                    |
|`training_fold[0-3].txt`               |storing training IDs and filenames                             |
|`testing_fold[0-3].txt`                |storing testing IDs and filenames                              |

### Training
Training is relatively slow and I think the distance map computation is time consuming. For training:
```
python SurfaceTrain.py
```
Then, logs and models folders will be generated:

| Folder/File                                   | Description                                          |
|:----------------------------------------------|:-----------------------------------------------------|
|**logs/**                                      |storing loss log csv file                             |
|`2019-12-09-loss.csv`                          |training and validation loss                          |
|**models/**                                    |storing model                                         |
|`unet-loss-0.7388-epoch0001-18-28-10.model`    |model state                                           |

### Testing
For testing:
```
python SurfaceOutput.py
```
This will generate an output folder, which including the predict testing results.

## Versions
The current version is 1.0. The current version is not well established, and the future version will come soon. 

## Concat Information
If you encounter any problems in using these codes, please open an issue in this repository. You may also contact **Kang Yang** (yangkang323@gmail.com).

Thanks for your interest! Have fun!


   
