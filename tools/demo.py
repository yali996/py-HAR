#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test_retina import im_detect
# from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('Person', 'bicycle', 'car', 'motorcycle','airplane','bus','train','truck','boat',
           'traffic light','fire hydrant','stop sign','parking meter','bench',
           'bird', 'cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',  
           'backpack','umbrella','handbag','tie','suitcase',
           'frisbee','Skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
           'bottle','wine glass','cup','fork','knife','spoon','bowl',
           'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
           'chair','couch','potted plant','bed','dining table','toilet',
           'tv','laptop','mouse','remote','keyboard','cell phone',
           'microwave','oven','toaster','sink','refrigerator',
           'book','clock','vase','scissors','teddy bear','hair drier','toothbrush')

NETS = {'res50': ('ResNet-50',
                  'resnet50_har_final.caffemodel'),
        'res101': ('ResNet-101',
                  'ResNet-101_har_final.caffemodel')}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    dets = im_detect(net, im)
    scores = dets[:,-1]
    boxes = dets[:,1:5] 
    clss = dets[:,0]
 
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.4
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[0:]):
        # cls_ind += 1 # because we skipped background
        keep = np.where((clss == cls_ind))[0]
        if len(keep) == 0:
            continue
        cls_boxes = boxes[keep,:]
        cls_scores = scores[keep]
        print "class:", cls, " max score: ", np.max(cls_scores,axis=None)
        # dets = np.hstack((cls_boxes, cls_scores[:,np.newaxis]))
        dets = np.hstack((cls_boxes, cls_scores[:,np.newaxis]))
        vis_detections(im, cls, dets, thresh=CONF_THRESH)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Retina-net demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [res101]',
                        choices=NETS.keys(), default='res101')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = parse_args()
    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'retina_end2end', 'test_hyr_att.prototxt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'retina_models',
                              NETS[args.demo_net][1])
    

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name) 

    plt.show()
