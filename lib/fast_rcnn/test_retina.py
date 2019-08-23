# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an imdb (image database)."""

from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
from fast_rcnn.nms_wrapper import nms
import cPickle
from utils.blob import im_list_to_blob
from utils.cython_bbox import bbox_overlaps
import scipy.io as sio
import os
import copy

def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if not cfg.TEST.HAS_RPN:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors

def im_detect(net, im):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, im_scales = _get_blobs(im, None)

    im_blob = blobs['data']
    blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)

    # reshape network inputs
    net.blobs['data'].reshape(*(blobs['data'].shape))
    net.blobs['im_info'].reshape(*(blobs['im_info'].shape))

    # do forward
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    forward_kwargs['im_info'] = blobs['im_info'].astype(np.float32, copy=False)
    blobs_out = net.forward(**forward_kwargs)
     
    S3 = net.blobs['spatt_P3_mask/1x1'].data
    S4 = net.blobs['spatt_P4_mask/1x1'].data
    S5 = net.blobs['spatt_P5_mask/1x1'].data
    S6 = net.blobs['spatt_P6_mask/1x1'].data
    S7 = net.blobs['spatt_P7_mask/1x1'].data
    print S3.max(axis=None), S3.min(axis=None), S3.mean(axis=None)
    print S4.max(axis=None), S4.min(axis=None), S4.mean(axis=None)
    print S5.max(axis=None), S5.min(axis=None), S5.mean(axis=None) 
    print S6.max(axis=None), S6.min(axis=None), S6.mean(axis=None)
    print S7.max(axis=None), S7.min(axis=None), S7.mean(axis=None)
    # sio.savemat('debug.mat', {'S3': S3, 'S4': S4, 'S5': S5, 'S6': S6, 'S7': S7 })    
    att_map = {'S3':S3, 'S4':S4, 'S5':S5, 'S6': S6, 'S7': S7 }
    
    ''' 
    offset_c5 = net.blobs['fpn_offset_conv5_1b'].data
    # print offset_c5
    print offset_c5.max(axis=None), offset_c5.min(axis=None)
    offset_c4 = net.blobs['fpn_offset_conv4_1b'].data
    # print offset_c4
    print offset_c4.max(axis=None), offset_c4.min(axis=None)
    # sio.savemat('debug_dc.mat', {'offset_c5': offset_c5, 'offset_c4': offset_c4 }) 
    '''
    '''
    FC3 = net.blobs['fpn_conv3x3_P3_CA4'].data
    FB3 = net.blobs['fpn_conv3x3_P3_BA4'].data
    FC4 = net.blobs['fpn_conv3x3_P4_CA4'].data
    FB4 = net.blobs['fpn_conv3x3_P4_BA4'].data
    FC5 = net.blobs['fpn_conv3x3_P5_CA4'].data
    FB5 = net.blobs['fpn_conv3x3_P5_BA4'].data
    FC6 = net.blobs['fpn_conv3x3_P6_CA4'].data
    FB6 = net.blobs['fpn_conv3x3_P6_BA4'].data
    FC7 = net.blobs['fpn_conv3x3_P7_CA4'].data
    FB7 = net.blobs['fpn_conv3x3_P7_BA4'].data
    CA = net.blobs['fpn_CA4_1x1_up'].data
    BA = net.blobs['fpn_BA4_1x1_up'].data
    print CA.max(axis=None), CA.min(axis=None),
    print BA.max(axis=None), BA.min(axis=None),
    sio.savemat('debug_ch.mat', {'CA': CA, 'BA': BA, 'FC3': FC3, 'FB3': FB3, 'FC4': FC4, 'FB4': FB4, 
                                 'FC5': FC5, 'FB5': FB5, 'FC6' : FC6, 'FB6' : FB6, 'FC7':FC7, 'FB7':FB7 }) 
    '''
    dets = blobs_out['dets']
    dets[:,1:5] /= im_scales[0]
    return dets, att_map
    # return dets
    # return dets, [offset_c5]


def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()

def mark_detections(im, dets, class_name=None, conf_thresh=0.4):
    # mark objects
    scores = dets[:,-1]
    boxes = dets[:,1:5]
    clss = dets[:,0]

    im2 = im.copy()
    # font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8) #Creates a font
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL
    # font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.8
    ind = np.where( scores >= conf_thresh )[0]
    for k in xrange(len(ind)):
        sx1 = int(boxes[ind[k],0])
        sy1 = int(boxes[ind[k],1])
        sx2 = int(boxes[ind[k],2])
        sy2 = int(boxes[ind[k],3])
        cls_ind = int(clss[ind[k]])
        
        if cls_ind in set([0]): # person
            color = (0,255,0)
        elif cls_ind in xrange(1,9): # vehicle
            color = (0,255,255)
        elif cls_ind in xrange(9,14): #outdoor 
            color = (255,255,0)
        elif cls_ind in xrange(14,24): #animal
            color = (0,255,122)
        elif cls_ind in xrange(24,29): #accesory
            color = (122,255,0)
        elif cls_ind in xrange(29,39): #sports
            color = (0,0,255)
        elif cls_ind in xrange(39,46): #kitchen
            color = (0,122,255)
        elif cls_ind in xrange(46,56): #food
            color = (122,0,255)
        elif cls_ind in xrange(56,62): #furniture
            color = (255,0,255)
        elif cls_ind in xrange(62,68): #electronic
            color = (255,0,0)
        elif cls_ind in xrange(68,73): #appliance
            color = (255,122,122)
        elif cls_ind in xrange(73,80): #indoor
            color = (255,56,56)

        cv2.rectangle(im2,(sx1,sy1),(sx2,sy2),color,2)
        text_size = cv2.getTextSize(class_name[cls_ind]+':{:.3f}'.format(scores[ind[k]]), font, font_scale, 1)
        cv2.rectangle(im2, (sx1, sy1-text_size[0][1]/2-1), (sx1+text_size[0][0], sy1++text_size[0][1]/2+1),(255,255,255),-1,4)
        cv2.putText(im2, class_name[cls_ind]+':{:.3f}'.format(scores[ind[k]]),(sx1 ,sy1+text_size[0][1]/2), font, font_scale, (0,0,0))
    cv2.addWeighted(im2, 0.8, im, 0.2, 0, im2)
    return im2


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

def test_net(net, imdb, max_per_image=100, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)
    # resimagespath = '/home/liyl/code/py-HAR/output/retina_end2end/coco_2014_minival/detection_images/'
    attmappath = '/home/liyl/code/py-HAR/output/retina_end2end/coco_2014_minival/attention_maps/'

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for i in xrange(num_images):
        # filter out any ground truth boxes

        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        dets, att_map = im_detect(net, im)
        _t['im_detect'].toc()

        _t['misc'].tic()
   
        if True:
            # im2 = mark_detections(im, dets,  imdb.classes[1:])
            # cv2.imwrite(os.path.join(resimagespath, os.path.split(imdb.image_path_at(i))[-1]), im2)
            im_name =  os.path.split(imdb.image_path_at(i))[-1]
            sio.savemat(os.path.join(attmappath, im_name[:-3]+'mat') , att_map)

        # skip j = 0, because it's the background class
        for j in xrange(1,imdb.num_classes):
            inds = np.where( (dets[:,0] == j-1)&(dets[:,-1] >= thresh))[0]
            cls_dets = dets[inds, 1:]
            if vis:
                vis_detections(im, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)

