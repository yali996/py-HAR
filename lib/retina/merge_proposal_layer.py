# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.nms_wrapper import nms, soft_nms

DEBUG = False

class MergeProposalLayer(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params.get('num_classes', 80)
        self._nms_thresh = layer_params.get('nms_thresh', 0.5)
        self._bbox_thresh = layer_params.get('bbox_thresh', 0.6)
        self._num_neighbors = layer_params.get('num_neighbors', 4)

        if DEBUG:
            print 'thresh: {}'.format(self._nms_thresh)

        # output blob: holds M detections, each is a 6-tuple
        # (c, x1, y1, x2, y2, score) specifying the class label c and a
        # rectangle (x1, y1, x2, y2) as well as the confidence score
        top[0].reshape(1, 6)


    def forward(self, bottom, top):
        # Algorithm:
        #
        # merge the detections from different levels in the pyramid 
        # apply NMS with threshold 0.5 
        # apply bounding box voting
        # return the detections attached with class labels and confidence scores 

        num_scales = len(bottom)      
        num_classes = self._num_classes         
        nms_thresh = self._nms_thresh        

        if DEBUG:
            print 'classes: {}, scales {}'.format(num_classes, num_scales)

        merge_dets = np.zeros((0, 6), dtype=np.float32) 
        for c in xrange(num_classes):
            all_dets = np.zeros((0, 6), dtype=np.float32)
            for i in xrange(num_scales):
                dets = bottom[i].data.reshape((-1,6))
                ind = np.where( dets[:,0] == c )[0]
                if len(ind) > 0:
                    all_dets = np.vstack((all_dets, dets[ind, :]))
            if all_dets.shape[0] > 1:    
                # keep = nms(all_dets[:,1:], nms_thresh)
                # merge_dets = np.vstack((merge_dets, all_dets[keep, :]))
                keep_dets = soft_nms(all_dets[:,1:], sigma=0.5, method = 2)
                C_F = c * np.ones((keep_dets.shape[0],1), dtype=np.float32)
                # keep_dets = np.hstack( (C_F, keep_dets) )
                # merge_dets = np.vstack((merge_dets, keep_dets))
                B_F, S_F = _bbox_voting( all_dets[:, 1:], keep_dets, self._num_neighbors, self._bbox_thresh )
                merge_dets = np.vstack((merge_dets, np.hstack((C_F, B_F, S_F[:,np.newaxis]))))
               
        top[0].reshape(*(merge_dets.shape))
        top[0].data[...] = merge_dets


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _bbox_voting(cls_dets, keep_dets, num_neighbors, bbox_thresh):
    """Bbox voting."""
    
    N = keep_dets.shape[0]
    Scores_F = np.zeros((N), dtype=np.float32)
    Boxes_F = np.zeros((N,4), dtype=np.float32)

    overlaps = bbox_overlaps(
               np.ascontiguousarray(keep_dets[:,:4], dtype=np.float),
               np.ascontiguousarray(cls_dets[:,:4], dtype=np.float))
 
    is_used = np.zeros((cls_dets.shape[0]), dtype=np.int32)
    for n in xrange(N):
        idx = np.where(overlaps[n,:] == 1.0)[0]
        is_used[idx] = -1
    for n in xrange( N ):
        J = np.where( (overlaps[n,:] >= bbox_thresh ) & (is_used == 0) )[0]
        if len(J) > 0:
            is_used[J] = 1
            order = np.argsort(-cls_dets[J,-1])
            num_fuse = np.minimum(num_neighbors, order.size)
            s = np.zeros((num_fuse + 1), dtype=np.float32)
            b = np.zeros((num_fuse + 1, 4), dtype=np.float32)
            s[0] = keep_dets[n,-1] 
            b[0,:] = keep_dets[n,:4] 
            for m in xrange(num_fuse):
                l = int(order[m])
                s[m+1] = cls_dets[J[l],-1]
                b[m+1,:] = cls_dets[J[l],:4]
            Scores_F[n] = np.sum(s, axis=None)/(num_neighbors + 1)
            Boxes_F[n,:] = np.dot(s, b)/ np.sum(s, axis=None)
        else:
            Scores_F[n] = keep_dets[n,-1]
            Boxes_F[n,:] = keep_dets[n,:4]
            
    return Boxes_F, Scores_F
