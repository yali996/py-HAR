#include <algorithm>
#include <vector>

#include "caffe/layers/sync_group_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SyncGroupNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  GroupNormParameter param = this->layer_param_.group_norm_param();
  group_num_ = param.group_num();
  eps_ = param.eps();
  
  channels_ = bottom[0]->shape(1);
  chip_num_ = int(channels_ / group_num_);

  num_ = bottom[0]->shape(0);
  CHECK_EQ(channels_ % group_num_,0);
}

template <typename Dtype>
void SyncGroupNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  CHECK_EQ(bottom.size(), top.size());
  for ( int n=0; n < bottom.size(); ++n )  
    top[n]->ReshapeLike(*bottom[n]);
 
  vector<int> sz;
  sz.push_back(num_ * group_num_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  buf_.Reshape(sz);

  spatial_dim_ = -1;
  sum_spatial_dim_ = 0;
  for (int n = 0; n < bottom.size(); ++n){ 
    sum_spatial_dim_ += int(bottom[n]->count()/(channels_*num_));
    spatial_dim_ = std::max( spatial_dim_, int(bottom[n]->count()/(channels_*num_)) );
  } 
  
  temp_.Reshape(num_, channels_, spatial_dim_, 1); // memory for saving temporal data
  x_norm_.Reshape(num_, channels_, sum_spatial_dim_, 1);
  
  // LOG(INFO) << spatial_dim_ << " " << sum_spatial_dim_; 
  sz[0] = spatial_dim_ * chip_num_;
  cube_sum_multiplier_.Reshape(sz);
  Dtype* cube_sum_multiplier_data = cube_sum_multiplier_.mutable_cpu_data();
  caffe_set(cube_sum_multiplier_.count(), Dtype(1.), cube_sum_multiplier_data);
}


#ifdef CPU_ONLY
STUB_GPU(SyncGroupNormLayer);
#endif

INSTANTIATE_CLASS(SyncGroupNormLayer);
REGISTER_LAYER_CLASS(SyncGroupNorm);
}  // namespace caffe
