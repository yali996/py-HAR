#include <algorithm>
#include <cmath>
#include <vector>
#include <functional>

#include "caffe/util/rpn.hpp"

/*using std::max;
using std::min;
using std::floor;
using std::ceil;
using std::round;
*/

namespace caffe {


template <typename Dtype>
__global__ void EnumerateAllAnchorsKernel(const int nthreads, const Dtype* anchors, const int anchor_dim, const int feat_stride, 
                                          const int height, const int width, Dtype * all_anchors ){
  // const int spatial_dim = height * width; // h, w is the size of feature map
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index % anchor_dim;
    const int s = index / anchor_dim;
    const int h = s / width;
    const int w = s % width;   

    const int top_idx = 4 * index;
    all_anchors[top_idx] = w * feat_stride + anchors[4*n]; 
    all_anchors[top_idx+1] = h * feat_stride + anchors[4*n+1]; 
    all_anchors[top_idx+2] = w * feat_stride + anchors[4*n+2]; 
    all_anchors[top_idx+3] = h * feat_stride + anchors[4*n+3]; 
  }
}


template <typename Dtype>
void EnumerateAllAnchorsGPU(const int nthreads, const Dtype* anchor_data, const int anchor_dim, const int feat_stride, 
                            const int height, const int width, Dtype * all_anchors ){
  // NOLINT_NEXT_LINE(whitespace/operators)
  EnumerateAllAnchorsKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>( nthreads, anchor_data, anchor_dim, feat_stride, 
      height, width, all_anchors );
  CUDA_POST_KERNEL_CHECK;
}

template
void EnumerateAllAnchorsGPU(const int nthreads, const float* anchor_data, const int anchor_dim, 
                            const int feat_stride, const int height, const int width, float * all_anchors);


template
void EnumerateAllAnchorsGPU(const int nthreads, const double* anchor_data, const int anchor_dim, 
                            const int feat_stride, const int height, const int width, double * all_anchors);



template <typename Dtype>
__global__ void BBoxOverlapKernel(const int nthreads, const Dtype* rois, const int roi_dim, const int num_rois,
                                  const Dtype* annos, const int anno_dim, const int num_annos, Dtype * overlaps ){
  // const int spatial_dim = height * width; // h, w is the size of feature map
  CUDA_KERNEL_LOOP(index, nthreads) {
    
    const int m = index / num_rois;
    const int n = index % num_rois;

    Dtype xmin1 = rois[n*roi_dim];
    Dtype ymin1 = rois[n*roi_dim + 1];
    Dtype xmax1 = rois[n*roi_dim + 2];
    Dtype ymax1 = rois[n*roi_dim + 3];

    Dtype xmin2 = annos[m*anno_dim];
    Dtype ymin2 = annos[m*anno_dim + 1];
    Dtype xmax2 = annos[m*anno_dim + 2];
    Dtype ymax2 = annos[m*anno_dim + 3];

    if( xmax2 < xmin1 || ymax2 < ymin1 || xmax1 < xmin2 || ymax1 < ymin2 )
        overlaps[index] = 0.0;
    else {
        Dtype x1 = max(xmin1, xmin2);
        Dtype y1 = max(ymin1, ymin2);
        Dtype x2 = min(xmax1, xmax2);
        Dtype y2 = min(ymax1, ymax2);
        overlaps[index] = ( x2-x1+1 )*(y2-y1+1)/( (xmax1-xmin1+1)*(ymax1-ymin1+1)+(xmax2-xmin2+1)*(ymax2-ymin2+1) - ( x2-x1+1 )*(y2-y1+1) );
    }
  }
}

template <typename Dtype>
void BBoxOverlapGPU(const int nthreads, const Dtype* rois, const int roi_dim, const int num_rois,                                
                    const Dtype* annos, const int anno_dim, const int num_annos, Dtype * overlaps){
  // NOLINT_NEXT_LINE(whitespace/operators)
  BBoxOverlapKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>( nthreads, rois, roi_dim, num_rois,  annos, anno_dim, num_annos, overlaps);
  CUDA_POST_KERNEL_CHECK;
}

template 
void BBoxOverlapGPU(const int nthreads, const float* rois, const int roi_dim, const int num_rois,                                                                         const float* annos, const int anno_dim, const int num_annos, float * overlaps);

template 
void BBoxOverlapGPU(const int nthreads, const double* rois, const int roi_dim, const int num_rois,                                                                         const double* annos, const int anno_dim, const int num_annos, double * overlaps);


template <typename Dtype>
__global__ void BBoxTransformKernel(const int nthreads, const Dtype* ex_rois, const Dtype* gt_rois, Dtype* bbox_targets){
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = 4 * index;
    Dtype ex_cx, ex_cy, ex_w, ex_h;
    Dtype gt_cx, gt_cy, gt_w, gt_h;
    ex_w = ex_rois[n+2] - ex_rois[n]+1;
    ex_h = ex_rois[n+3] - ex_rois[n+1]+1;
    ex_cx = ex_rois[n] + ex_w/2;
    ex_cy = ex_rois[n+1] + ex_h/2;

    gt_w = gt_rois[n+2] - gt_rois[n]+1;
    gt_h = gt_rois[n+3] - gt_rois[n+1]+1;
    gt_cx = gt_rois[n] + gt_w/2;
    gt_cy = gt_rois[n+1] + gt_h/2;

    bbox_targets[n] = (gt_cx - ex_cx)/ex_w;
    bbox_targets[n+1] = (gt_cy - ex_cy)/ex_h;
    bbox_targets[n+2] = log(gt_w/ex_w);
    bbox_targets[n+3] = log(gt_h/ex_h);
  }
} 

template <typename Dtype>
void BBoxTransformGPU(const int nthreads, const Dtype * ex_rois, const Dtype * gt_rois, Dtype * bbox_targets ){
  // NOLINT_NEXT_LINE(whitespace/operators)
  BBoxTransformKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>( nthreads, ex_rois, gt_rois, bbox_targets );
  CUDA_POST_KERNEL_CHECK;
}

template
void BBoxTransformGPU(const int nthreads, const float * ex_rois, const float * gt_rois, float * bbox_targets);

template
void BBoxTransformGPU(const int nthreads, const double * ex_rois, const double * gt_rois, double * bbox_targets);



template <typename Dtype>
__global__ void BBoxTransformInvKernel(const int nthreads,  const Dtype * boxes, const Dtype * deltas, const int M, Dtype* pred_boxes){
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / M;
    const int m = index % M;
    Dtype cx, cy, w, h;
    Dtype px, py, pw, ph;
    
    w = boxes[4*n+2] - boxes[4*n]+1.0;
    h = boxes[4*n+3] - boxes[4*n+1]+1.0;
    cx = boxes[4*n] + 0.5*w - 1.0;
    cy = boxes[4*n+1] + 0.5*h - 1.0;

    px = deltas[(n*M+m)*4]*w+cx;
    py = deltas[(n*M+m)*4+1]*h+cy;
    pw = exp(deltas[(n*M+m)*4+2]) * w;
    ph = exp(deltas[(n*M+m)*4+3]) * h;

    pred_boxes[(n*M+m)*4] = px - 0.5 * pw;
    pred_boxes[(n*M+m)*4+1] = py - 0.5 * ph;
    pred_boxes[(n*M+m)*4+2] = px + 0.5 * pw;
    pred_boxes[(n*M+m)*4+3] = py + 0.5 * ph;
  }
}


template <typename Dtype>
void BBoxTransformInvGPU(const int nthreads, const Dtype * boxes, const Dtype * box_deltas, const int num_classes, Dtype* pred_boxes ){
  // NOLINT_NEXT_LINE(whitespace/operators)
  BBoxTransformInvKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>( nthreads, boxes, box_deltas, num_classes, pred_boxes );
  CUDA_POST_KERNEL_CHECK;
}

template
void BBoxTransformInvGPU(const int nthreads, const float * boxes, const float * box_deltas, const int num_classes, float* pred_boxes);

template
void BBoxTransformInvGPU(const int nthreads, const double * boxes, const double * box_deltas, const int num_classes, double* pred_boxes);


}

	


