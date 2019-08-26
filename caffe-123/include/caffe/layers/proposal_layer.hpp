#ifndef CAFFE_PROPOSAL_LAYER_HPP_
#define CAFFE_PROPOSAL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ProposalLayer : public Layer<Dtype> {
 public:
  explicit ProposalLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Proposal"; }

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int MaxTopBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
      for (int i = 0; i < propagate_down.size(); ++i) {
          if (propagate_down[i]) { NOT_IMPLEMENTED; }
      }
  }

  int channels_;
  int height_;
  int width_;
  int feat_stride_;
  int keep_cls_info_;
  vector<Dtype> scales_;
  vector<Dtype> anchors_;
  float nms_thresh_, thresh_;
  int pre_nms_topN, post_nms_topN;
  int min_size_; 
  bool normalize_targets_;
  vector<Dtype> target_mean, target_stds;
};


}  // namespace caffe

#endif  // CAFFE_PROPOSAL_LAYER_HPP_
