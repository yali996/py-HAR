#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/upsample_layer.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
  __global__ void UpsampleForward(const int nthreads, int in_w, int in_h,
      int out_w, int out_h, int scale_w, int scale_h, 
      const Dtype* bottom_data, Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int offset = index / (out_w * out_h) * in_w * in_h;
      int idx_w = ( index % out_w ) / scale_w;
      int idx_h = ( (index / out_w ) % out_h) / scale_h;
      top_data[index] = bottom_data[offset + idx_h * in_w + idx_w ];
    }
  }

template <typename Dtype>
void UpsampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
//  const Dtype* bottom_mask = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
  const int bottom_count = bottom[0]->count();
  const int top_count = top[0]->count();
  UpsampleForward<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
      top_count, bottom[0]->width(), bottom[0]->height(), 
      top[0]->width(), top[0]->height(), scale_w_, scale_h_, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
  __global__ void UpsampleBackward(const int nthreads, int in_w, int in_h,
      int out_w, int out_h, int scale_w, int scale_h,
      const Dtype* top_diff, Dtype* bottom_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int offset = index / (in_w * in_h) * out_w * out_h;
      int idx_x = index % in_w;
      int idx_y = ( index / in_w ) % in_h;
      bottom_diff[index] = 0;
      int start_w = idx_x * scale_w;
      int end_w = ( idx_x + 1 ) * scale_w;
      int start_h = idx_y * scale_h;
      int end_h = ( idx_y + 1 ) * scale_h;
      end_w = min( end_w, out_w );
      end_h = min( end_h, out_h );
      for ( int i = start_w; i < end_w; ++i)
        for( int j = start_h; j < end_h; j++){
          bottom_diff[index] += top_diff[offset + j*out_w + i];
      }
    }
  }

template <typename Dtype>
void UpsampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
//    const Dtype* bottom_mask = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    caffe_gpu_set(bottom_count, Dtype(0.), bottom_diff);
    UpsampleBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_count, bottom[0]->width(), bottom[0]->height(), 
        top[0]->width(), top[0]->height(), scale_w_, scale_h_, top_diff, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(UpsampleLayer);


}  // namespace caffe

