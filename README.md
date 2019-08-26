# py-HAR
This is the code for HAR-Net, which is a single-stage object detection method.

# HAR-Net: Joint Learning of Hybrid Attention for Single-stage Object Detection


# Introduction

HAR-Net is a single-stage object detection method with hybrid attention mechanism, such as spatial attention, channel attention and aligned attention. 
This is a Python implementation, which is modified from the py-faster-rcnn. 

This code is only released for academic use. 

# Setting up py-HAR

## Step 0: Clone this repo and copy files to the required dictionary.
`git clone https://github.com/yali996/py-HAR.git`

## Step 1: Compile the master branch of caffe
Checkout the master branch of Caffe and compile it on your machine. Make sure that Caffe must be built with support for Python layers!

## Step 2: Compile the nms module
`cd lib`
`make`

# Testing
For testing the models, please first download the models with the links below.

ResNet-50: [@BaiduYunDrive]{https://pan.baidu.com/s/1UysG_E_CvNt206XA8JhJmQ}, train on COCO 115k dataset, with 38.9% AP on COCO 5k validation dataset.

ResNet-101: [@BaiduYunDrive]{https://pan.baidu.com/s/14LzWyGzxzPlmthKGqExn9g}, train on COCO 115k dataset, with 40.6% AP on COCO 5k validation dataset.

More models is coming soon.
 
# Training
The traning scrip file is coming soon.
