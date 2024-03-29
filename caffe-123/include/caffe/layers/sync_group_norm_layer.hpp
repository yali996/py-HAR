#ifndef CAFFE_SYNC_GROUPNORM_LAYER_HPP_
#define CAFFE_SYNC_GROUPNORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


template <typename Dtype>
class SyncGroupNormLayer : public Layer<Dtype> {
 public:
  explicit SyncGroupNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SyncGroupNorm"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 10; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 10; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){;};
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){;};
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> mean_, variance_, buf_;
  int group_num_;
  int channels_;
  int chip_num_;
  int num_;
  int spatial_dim_, sum_spatial_dim_;
  Dtype eps_;
  
  Blob<Dtype> x_norm_;
  Blob<Dtype> temp_;
  // extra temporarary variables is used to carry out sums/broadcasting
  // using BLAS
  Blob<Dtype> cube_sum_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_SYNC_GROUPNORM_LAYER_HPP_
