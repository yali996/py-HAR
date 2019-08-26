#ifndef CAFFE_UTIL_RPN_FUNCTIONS_H_
#define CAFFE_UTIL_RPN_FUNCTIONS_H_


#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
void rpn_generate_default_anchors_ss( int scale, vector<Dtype> &anchors);

template<typename Dtype>
void rpn_generate_default_anchors( vector<int> scales, vector<Dtype> &anchors);

// template <typename Dtype>
// void rpn_generate_anchors(const vector<Dtype> base_anchor, const vector<Dtype> ratios, int scale_min, int num_scales, vector<Dtype> &anchors);

template <typename Dtype>
void rpn_generate_anchors(const vector<Dtype> base_anchor, const vector<Dtype> ratios, const vector<Dtype> scales, vector<Dtype> &anchors);

template<typename Dtype>
void generate_default_dense_anchors(const int feat_stride, vector<Dtype> &anchors);
template<typename Dtype>
void generate_dense_anchors(const int feat_stride, const vector<Dtype> scales, vector<Dtype> &anchors);

template<typename Dtype>
void rpn_generate_dense_anchors(const vector<Dtype> base_anchor, const vector<Dtype> scales, vector<Dtype> &anchors);

template <typename Dtype>
void rpn_enumerate_all_anchors(const vector<Dtype> anchors, int feat_stride, int h, int w, Dtype* all_anchors );

template <typename Dtype>
void rpn_enumerate_all_anchors(const Dtype* anchors, const int A, const int feat_stride, const int h, const int w, Dtype * all_anchors );


template <typename Dtype>
void rpn_filter_anchors( vector<Dtype> all_anchors, const Dtype * im_info, int allowed_border, vector<int> &inds_inside );

template <typename Dtype>
Dtype bbox_overlap( Dtype xmin1, Dtype ymin1, Dtype xmax1, Dtype ymax1, Dtype xmin2, Dtype ymin2, Dtype xmax2, Dtype ymax2 );

template<typename Dtype>
Dtype euclidean_feature_dist( const Dtype * ftrs_a, const Dtype * ftrs_b, int dim );

template <typename Dtype>
void rpn_normalize_targets( vector<Dtype> &targets, vector<Dtype> mean, vector<Dtype> stds );

template<typename Dtype>
void normalize_targets( Dtype* targets, const int M, const vector<Dtype> mean, const vector<Dtype> stds );

template <typename Dtype>
void rpn_unnormalize_targets( vector<Dtype> &targets, vector<Dtype> mean, vector<Dtype> stds );

void generate_random_sequence(int max_num, int num, vector<int> & rand_inds );

template <typename Dtype>
void rpn_bbox_transform(vector<Dtype> ex_rois, vector<Dtype> gt_rois, vector<Dtype> &bbox_targets );

template <typename Dtype>
void bbox_transform(const Dtype * ex_rois, const Dtype * gt_rois, const int num_rois, Dtype * bbox_targets );

template <typename Dtype>
void rpn_bbox_transform_inv(vector<Dtype> boxes, vector<Dtype> deltas, vector<Dtype> & pred_boxes);

template <typename Dtype>
void rpn_bbox_transform_inv(const Dtype * boxes, const Dtype * deltas, const int N, const int M, vector<Dtype> & pred_boxes);

template <typename Dtype>
void rpn_clip_boxes(vector<Dtype> & boxes, const Dtype * im_shape );

template <typename Dtype>
void clip_boxes(Dtype* boxes, const int M, const Dtype * im_shape );

template <typename Dtype>
void rpn_filter_small_proposals( vector<Dtype> proposals, Dtype min_size, vector<int> & keep );

/*template <typename Dtype>
void nms(const vector<Dtype> proposals, const vector<Dtype> scores, const vector<Dtype> features,
		         const float nms_thresh, const float sim_thresh, int feature_dim, vector<int> & keep ); */

template <typename Dtype>
void nms(const Dtype * proposals, const Dtype * scores, const Dtype * features,
         const int proposals_dim, const int scores_dim, const int feature_dim,
         const int N, const float nms_thresh, const float sim_thresh, vector<int> & keep );


template <typename Dtype>
void nms(const vector<Dtype> proposals, const vector<Dtype> scores, const float nms_thresh, vector<int> & keep );

/* template <typename Dtype>
void allocate_bags_with_nms(const vector<Dtype> proposals, const vector<Dtype> scores, const float nms_thresh,
                           vector<int> keep, vector<int> &bag_index );
*/

template <typename Dtype>
void get_refined_boxes(const vector<Dtype> proposals, const vector<Dtype> cls_probs, const float ovlp_thresh, 
			              vector<int> keep, vector<Dtype> &reg_boxes );

template <typename Dtype>
void overlap_distance(const Dtype * cls_boxes, // detected boxes
                      const Dtype * gt_boxes,  // ground truth boxes
                      const int M, // number of detections
                      const int N, // number of ground truths
                      const int dim, // dim for cls_boxes, gt_boxes
                      Dtype * D );

template <typename Dtype>
void EnumerateAllAnchorsGPU(const int nthreads, const Dtype* anchor_data, const int anchor_dim, const int feat_stride,
                            const int height, const int width, Dtype * all_anchors );

template <typename Dtype>
void BBoxOverlapGPU(const int nthreads, const Dtype* rois, const int roi_dim, const int num_rois,                                                                         const Dtype* annos, const int anno_dim, const int num_annos, Dtype * overlaps);

template <typename Dtype>
void BBoxTransformGPU(const int nthreads, const Dtype * ex_rois, const Dtype * gt_rois, Dtype * bbox_targets );

template <typename Dtype>
void BBoxTransformInvGPU(const int nthreads, const Dtype * boxes, const Dtype * box_deltas, const int num_classes, Dtype* pred_boxes );


}  // namespace caffe

#endif  // 
