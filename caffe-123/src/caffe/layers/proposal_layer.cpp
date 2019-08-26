#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rpn.hpp"
#include "caffe/layers/proposal_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ProposalParameter rpn_proposal_param = this->layer_param_.proposal_param();
  CHECK_GT(rpn_proposal_param.feat_stride(), 0)
      << "feat_stride must be > 0";
  CHECK_GT(rpn_proposal_param.scale_min(), 0)
      << "scales must be > 0";
  feat_stride_ = rpn_proposal_param.feat_stride();
  nms_thresh_ = rpn_proposal_param.nms_thresh();
  thresh_ = rpn_proposal_param.thresh();
  pre_nms_topN = rpn_proposal_param.pre_nms_top_k();
  post_nms_topN = rpn_proposal_param.post_nms_top_k();
  min_size_ = rpn_proposal_param.min_size();
  keep_cls_info_ = rpn_proposal_param.keep_cls_info();

  for(int i=0; i<rpn_proposal_param.scales_size(); i++)
     scales_.push_back(rpn_proposal_param.scales(i));

  normalize_targets_ = rpn_proposal_param.normalize_targets();
  const RPNTargetNorm& target_norm = rpn_proposal_param.norm();
  CHECK_EQ(target_norm.mean_size(), target_norm.std_size() )
       << "dim of mean and stds must be equal";
  const int target_axes = target_norm.mean_size();
  for(int i =0; i < target_axes; i++){
      target_mean.push_back(target_norm.mean(i));
      target_stds.push_back(target_norm.std(i));
  }

  // rpn_generate_dense_anchors(scales_, anchors_);
  generate_default_dense_anchors(feat_stride_, anchors_);
  LOG(INFO) << "anchor scales: " << anchors_.size();
}


template <typename Dtype>
void ProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  const int num_anchors = anchors_.size()/4;
  const int num_classes = bottom[0]->width()/num_anchors;
  vector<int> top_shape;
  vector<int> score_shape;

  if(pre_nms_topN > 0){
     top_shape.push_back(bottom[0]->num() * pre_nms_topN);
     score_shape.push_back(bottom[0]->num() * pre_nms_topN);
  }
  else{
     top_shape.push_back(bottom[0]->num());
     score_shape.push_back(bottom[0]->num());
  }
  top_shape.push_back(6);
  score_shape.push_back(num_classes);
  top[0]->Reshape(top_shape);
  if(top.size()>1){
     top[1]->Reshape(score_shape);
  }
}

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//  const Dtype* scores_data = bottom[0]->cpu_data();
//  const Dtype* bbox_deltas_data = bottom[1]->cpu_data();
//  const Dtype* im_info = bottom[2]->cpu_data(); 
  const int num_anchors = anchors_.size()/4;
  const int num_classes = bottom[0]->width()/num_anchors;
  
  height_ = bottom[0]->channels();
  width_ = bottom[0]->height();
 
  vector<Dtype> all_anchors(height_*width_*num_anchors*4,0);
  rpn_enumerate_all_anchors(anchors_, feat_stride_, height_, width_, &all_anchors[0] );

  vector<Dtype> all_dets;
 // if (!all_dets.empty())
 //  all_dets.clear();
  vector<int> all_keep_inds;
//  vector<Dtype> all_scores;

  int num_reserved_rois = 0;
  for( int batch_ind = 0; batch_ind < bottom[0]->num(); ++batch_ind ){
    const Dtype* scores = bottom[0]->cpu_data() + bottom[0]->offset(batch_ind);
    const Dtype* bbox_deltas = bottom[1]->cpu_data() + bottom[1]->offset(batch_ind);
    const Dtype* im_info = bottom[2]->cpu_data() + bottom[2]->offset(batch_ind); 
    const int num_all_proposals = num_anchors * height_ *width_;
 /*  vector<Dtype> bbox_deltas(num_anchors*height_*width_*4,0); 
    if(normalize_targets_)
      rpn_unnormalize_targets(bbox_deltas, target_mean, target_stds);
 */
 
    vector<Dtype> proposals(num_all_proposals*4, 0);
//    LOG(INFO) << num_all_proposals << " " << num_anchors << " " << height_ << " " << width_;   
    
    rpn_bbox_transform_inv(&all_anchors[0], bbox_deltas, num_all_proposals, 1, proposals);
    rpn_clip_boxes( proposals, im_info );
   //  LOG(INFO) << num_all_proposals << " " << num_anchors << " " << height_ << " " << width_;   
  
    // 3. remove predicted boxes with either height or width < 
    vector<int> keep;
    rpn_filter_small_proposals( proposals, min_size_*im_info[2], keep );
 
    // 4. sort all (proposal, score) pairs by score from highest to lowest
    // 5. take top pre_nms_topN (e.g. 6000) 
    std::vector<std::pair<Dtype, int> > S(keep.size()*num_classes);
    for (int cls_ind = 0; cls_ind<num_classes; cls_ind++ ){ 
      // std::vector<std::pair<Dtype, int> > S(keep.size());
      for(int j=0; j<keep.size(); j++){
        S[j*num_classes+cls_ind] = std::make_pair(
                                     *(scores + keep[j]*num_classes + cls_ind), 
                                     keep[j]*num_classes+cls_ind);
      }
    }
    
    int num_cls_dets = pre_nms_topN > 0 ? min(static_cast<int>(keep.size()),pre_nms_topN) : keep.size();
    std::partial_sort(S.begin(), S.begin()+num_cls_dets - 1, S.end(), std::greater<std::pair<Dtype, int> >());
    for (int j=0; j<num_cls_dets; ++j){
        if (S[j].first < thresh_ && j > 0 )
          break;
        int cls_ind = S[j].second % num_classes;
        int anchor_ind = S[j].second / num_classes;
        if (keep_cls_info_)
           all_dets.push_back(static_cast<Dtype>(cls_ind)); // cls 
        else
           all_dets.push_back(static_cast<Dtype>(0)); // cls 
        all_dets.push_back(proposals[4*anchor_ind]); // / im_info[2]);  // xmin
        all_dets.push_back(proposals[4*anchor_ind+1]); // / im_info[2]); // ymin 
        all_dets.push_back(proposals[4*anchor_ind+2]); // / im_info[2]);  // xmax
        all_dets.push_back(proposals[4*anchor_ind+3]); // / im_info[2]);  // ymax
        all_dets.push_back(S[j].first); // cls_score 
        
        all_keep_inds.push_back(anchor_ind); // cls_score 
      
        // all_scores.resize((num_reserved_rois+1) * num_classes);
        // caffe_copy(num_classes, scores+anchor_ind*num_classes, &all_scores[num_reserved_rois*num_classes]);
        num_reserved_rois += 1;
    }

  // 6. apply nms (e.g. threshold = 0.7)
  // 7. take after_nms_topN (e.g. 300)
  // 8. return the top proposals (-> RoIs top)  
    if (nms_thresh_ > 0 && nms_thresh_ < 1){
       vector<Dtype> cls_boxes(num_reserved_rois*4);
       vector<Dtype> cls_scores(num_reserved_rois);
       for (int i=0; i < num_reserved_rois; ++i){
          caffe_copy(4, &all_dets[6*i+1], &cls_boxes[4*i] ); 
          caffe_copy(1, &all_dets[6*i+5], &cls_scores[i] ); 
       }
       vector<int> keep_nms;
       // LOG(INFO) << nms_thresh_;
       nms(cls_boxes, cls_scores, nms_thresh_, keep_nms );
       if(post_nms_topN >0 && keep_nms.size()> post_nms_topN)
          keep.erase(keep_nms.begin()+post_nms_topN, keep_nms.end()); 
     
       vector<int> temp(all_keep_inds); 
       all_dets.resize(keep_nms.size()*6);
       all_keep_inds.resize(keep_nms.size());
       for (int i=0; i<keep_nms.size(); ++i){
          all_dets[6*i] = Dtype(0);
          caffe_copy(4, &cls_boxes[4*keep_nms[i]], &all_dets[6*i+1]);  
          caffe_copy(1, &cls_scores[keep_nms[i]], &all_dets[6*i+5]);  
          all_keep_inds[i] = temp[keep_nms[i]];
       } 
       // LOG(INFO) << num_reserved_rois << " " << keep_nms.size(); 
     }
  }

  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = all_dets.size();
  const int top_num = all_dets.size()/6;

  // LOG(INFO) << top_num;
  vector<int> top_shape;
  top_shape.push_back(top_num);
  top_shape.push_back(6);
  top[0]->Reshape(top_shape);
  caffe_copy(top_count, &all_dets[0], top_data);
  
  if(top.size()>1){
    vector<int> score_shape;
    score_shape.push_back(top_num);
    score_shape.push_back(num_classes);
    top[1]->Reshape(score_shape);
    Dtype* top_scores = top[1]->mutable_cpu_data();
    for (int i=0; i < all_keep_inds.size(); ++i){
       caffe_copy(num_classes, bottom[0]->cpu_data()+all_keep_inds[i]*num_classes, top_scores+i*num_classes);
    }
  }
}

/*
#ifdef CPU_ONLY
STUB_GPU_FORWARD(ProposalLayer, Forward);
#endif
*/

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);

}  // namespace caffe
