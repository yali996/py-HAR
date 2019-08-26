#include <algorithm>
#include <vector>

#include "caffe/layers/sync_group_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SyncGroupNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
   
  // compute mean
  // sum_spatial_dim_ = 0;
  caffe_gpu_set(mean_.count(), Dtype(0.), mean_.mutable_gpu_data());
  for( int n=0; n < bottom.size(); n++ ){
    const Dtype* bottom_data = bottom[n]->gpu_data();
    Dtype* top_data = top[n]->mutable_gpu_data();
    
    if (bottom[n] != top[n]) {
      caffe_copy(bottom[n]->count(), bottom_data, top_data);
    }

    spatial_dim_ = int( bottom[n]->count()/(channels_*num_) );
    // sum_spatial_dim_ += spatial_dim_;
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * group_num_, chip_num_ * spatial_dim_,
        1. / ( chip_num_ * sum_spatial_dim_), bottom_data, cube_sum_multiplier_.gpu_data(), 1.0,
        mean_.mutable_gpu_data());
    // caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * group_num_, chip_num_ * spatial_dim_,
    //    1. / (chip_num_ * spatial_dim_), bottom_data, cube_sum_multiplier_.gpu_data(), 1.0,
    //    mean_.mutable_gpu_data());
  }
  // caffe_gpu_scal(mean_.count(), Dtype(1.0 / sum_spatial_dim_), mean_.mutable_gpu_data());
  
  // subtract mean
  for( int n=0; n < bottom.size(); n++ ){
    const Dtype* bottom_data = bottom[n]->gpu_data();
    Dtype* top_data = top[n]->mutable_gpu_data();
    spatial_dim_ = int( bottom[n]->count()/(channels_*num_) );
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,num_ * group_num_,
        chip_num_ * spatial_dim_, 1, -1., mean_.gpu_data(),
        cube_sum_multiplier_.gpu_data(), 1., top_data);
  } 
 
  // compute variance using var(X) = E((X-EX)^2)
  caffe_gpu_set(variance_.count(), Dtype(0.), variance_.mutable_gpu_data());
  for( int n=0; n < bottom.size(); n++ ){
    const Dtype* bottom_data = bottom[n]->gpu_data();
    Dtype* top_data = top[n]->mutable_gpu_data();
  
    // compute variance using var(X) = E((X-EX)^2)
    caffe_gpu_mul(top[n]->count(), top[n]->gpu_data(), top[n]->gpu_data(), temp_.mutable_gpu_data() );  // (X-EX)^2
    
    spatial_dim_ = int( bottom[n]->count()/(channels_*num_) );
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * group_num_, chip_num_ * spatial_dim_,
        1. / (chip_num_ * sum_spatial_dim_), temp_.gpu_data(),
        cube_sum_multiplier_.gpu_data(), 1.0, variance_.mutable_gpu_data());
  }
  // caffe_gpu_scal(variance_.count(), Dtype(1.0 / sum_spatial_dim_), variance_.mutable_gpu_data());
 
   // normalize variance
  caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
  caffe_gpu_sqrt(variance_.count(), variance_.gpu_data(), variance_.mutable_gpu_data());

  // div variance    
  int offset = 0;
  for( int n=0; n < bottom.size(); n++ ){
    const Dtype* bottom_data = bottom[n]->gpu_data();
    Dtype* top_data = top[n]->mutable_gpu_data();
    // div variance    
    spatial_dim_ = int( bottom[n]->count()/(channels_*num_) );
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * group_num_,
      chip_num_ * spatial_dim_, 1, 1., variance_.gpu_data(),
      cube_sum_multiplier_.gpu_data(), 0., temp_.mutable_gpu_data());

    caffe_gpu_div(top[n]->count(), top_data, temp_.gpu_data(), top_data);
  
    // TODO(cdoersch): The caching is only needed because later in-place layers
    //                 might clobber the data.  Can we skip this if they won't?
    caffe_copy(top[n]->count(), top_data, x_norm_.mutable_gpu_data() + offset);
    offset += bottom[n]->count();  
  }
}

template <typename Dtype>
void SyncGroupNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  int offset = 0;
  caffe_gpu_set(mean_.count(), Dtype(0.), mean_.mutable_gpu_data()); 
  for( int n=0; n < bottom.size(); n++ ){
    const Dtype* top_data = x_norm_.gpu_data() + offset;
    const Dtype* top_diff;
    Dtype* bottom_diff = bottom[n]->mutable_gpu_diff();

    if (bottom[n] != top[n]) {
      top_diff = top[n]->gpu_diff();
    } else {
      caffe_copy(top[n]->count(), top[n]->gpu_diff(), x_norm_.mutable_gpu_diff() + offset);
      top_diff = x_norm_.gpu_diff() + offset;
    }
 
    // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
    //
    // dE(Y)/dX =
    //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
    //     ./ sqrt(var(X) + eps)
    //
    // where \cdot and ./ are hadamard product and elementwise division,
    // respectively, dE/dY is the top diff, and mean/var/sum are all computed
    // along all dimensions except the channels dimension.  In the above
    // equation, the operations allow for expansion (i.e. broadcast) along all
    // dimensions except the channels dimension where required.
 
    spatial_dim_ = int( bottom[n]->count()/(channels_*num_) );
    // sum(dE/dY \cdot Y)
    caffe_gpu_mul(bottom[n]->count(), top_data, top_diff, bottom_diff);
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * group_num_, chip_num_ * spatial_dim_, 1.,
        bottom_diff, cube_sum_multiplier_.gpu_data(), 1.0,
        mean_.mutable_gpu_data());
    offset += bottom[n]->count();  
  }

  offset = 0; 
  for( int n=0; n < bottom.size(); n++ ){
    const Dtype* top_data = x_norm_.gpu_data() + offset;
    const Dtype* top_diff = (bottom[n] == top[n]) ? x_norm_.gpu_diff() + offset : top[n]->gpu_diff();
    Dtype* bottom_diff = bottom[n]->mutable_gpu_diff();
 
    // reshape (broadcast) the above
    spatial_dim_ = int( bottom[n]->count()/(channels_*num_) );
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * group_num_,
        chip_num_ * spatial_dim_, 1, 1., mean_.gpu_data(),
        cube_sum_multiplier_.gpu_data(), 0., bottom_diff);

    // sum(dE/dY \cdot Y) \cdot Y
    caffe_gpu_mul(bottom[n]->count(), top_data, bottom_diff, bottom_diff);
    offset += bottom[n]->count();  
  }

  // sum(dE/dY)
  offset = 0; 
  sum_spatial_dim_ = 0;
  caffe_gpu_set(mean_.count(), Dtype(0.), mean_.mutable_gpu_data()); 
  for( int n=0; n < bottom.size(); n++ ){
    const Dtype* top_data = x_norm_.gpu_data() + offset;
    const Dtype* top_diff = (bottom[n] == top[n]) ? x_norm_.gpu_diff() + offset : top[n]->gpu_diff();
    Dtype* bottom_diff = bottom[n]->mutable_gpu_diff();
  
    // sum(dE/dY)
    spatial_dim_ = int( bottom[n]->count()/(channels_*num_) );
    caffe_gpu_gemv<Dtype>(CblasNoTrans, num_ * group_num_, chip_num_ * spatial_dim_, 1.,
        top_diff, cube_sum_multiplier_.gpu_data(), 1.0,
        mean_.mutable_gpu_data());
    
    sum_spatial_dim_ += spatial_dim_;
    offset += bottom[n]->count();  
  } 
 
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  offset = 0; 
  for( int n=0; n < bottom.size(); n++ ){
    const Dtype* top_data = x_norm_.gpu_data() + offset;
    const Dtype* top_diff = (bottom[n] == top[n]) ? x_norm_.gpu_diff() + offset : top[n]->gpu_diff();
    Dtype* bottom_diff = bottom[n]->mutable_gpu_diff();
    // reshape (broadcast) the above to make
    // sum(dE/dY) + sum(dE/dY \cdot Y) \cdot Y
    spatial_dim_ = int( bottom[n]->count()/(channels_*num_) );
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * group_num_,
        chip_num_ * spatial_dim_, 1, 1., mean_.gpu_data(),
        cube_sum_multiplier_.gpu_data(), 1., bottom_diff);

    // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
    caffe_gpu_axpby(bottom[n]->count(), Dtype(1), top_diff,
        Dtype(-1. / (chip_num_ * sum_spatial_dim_)), bottom_diff);

    // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
    // pass.
    caffe_gpu_div(bottom[n]->count(), bottom_diff, temp_.gpu_data(), bottom_diff);
    offset += bottom[n]->count(); 
  }
}

 

INSTANTIATE_LAYER_GPU_FUNCS(SyncGroupNormLayer);


}  // namespace caffe




