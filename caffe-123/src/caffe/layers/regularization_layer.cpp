#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/regularization_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RegularizationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  K_ = bottom[0]->num();
  N_ = bottom[0]->count()/K_;
  // Initialize the weights
  /*
  vector<int> weight_shape(2);
  if (transpose_) {
    weight_shape[0] = K_;
    weight_shape[1] = N_;
  } else {
    weight_shape[0] = N_;
    weight_shape[1] = K_;
  } */
  top[0]->Reshape(K_,K_,1,1);
  if( top.size() > 1 )
      top[1]->Reshape(K_,K_,1,1);
}

template <typename Dtype>
void RegularizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(K_,K_,1,1);
  if( top.size() > 1 )
      top[1]->Reshape(K_,K_,1,1);
}

template <typename Dtype>
void RegularizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, K_, K_, N_, (Dtype)1.,
      bottom_data, bottom_data, (Dtype)0., top_data);
  for( int i = 0; i < K_; i++ )
      top_data[i*K_+i] = Dtype(0.0);   

  if(top.size() > 1 ){
     Dtype* top_target = top[1]->mutable_cpu_data();
     caffe_set(top[1]->count(), Dtype(0), top_target );
     for( int i = 0; i < K_; i++ )
         top_target[i*K_+i] = Dtype(1.0);   
  }
}

template <typename Dtype>
void RegularizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    
    if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, K_, N_, K_, (Dtype)2.,
      top_diff, bottom_data, (Dtype)0., bottom_diff);

/*    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    for(int i=0; i<K_; i++ )
        caffe_cpu_gemv<Dtype>(CblasTrans, K_, N_, Dtype(2.0), 
                              bottom_data, top_diff+i*K_,
                              Dtype(1.0), bottom_diff); */

    // Gradient with respect to bottom data
    /* if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } */
  }
}

/* ifdef CPU_ONLY
STUB_GPU(RegInnerProductLayer);
#endif */

INSTANTIATE_CLASS(RegularizationLayer);
REGISTER_LAYER_CLASS(Regularization);

}  // namespace caffe
