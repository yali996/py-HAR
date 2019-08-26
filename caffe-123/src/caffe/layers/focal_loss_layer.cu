#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void FullFocalLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* prob_data, 
          const Dtype* label, Dtype* loss,
          const float alpha, const float gamma,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else if( label_value > 0 ){
      loss[index] = - alpha * pow( Dtype(1.0)-prob_data[index], Dtype(gamma)) *
                 ( input_data[index] * (input_data[index] < 0) - log(1 + exp(input_data[index]
                  - 2 * input_data[index] * (input_data[index] >= 0))) );
      counts[index] = 1;
    }
    else{
      loss[index] = -(1.0 - alpha) * pow(prob_data[index], Dtype(gamma)) *
                  ( - input_data[index] * (input_data[index] >= 0)
                    - log(1 + exp(input_data[index] - 2 * input_data[index] * (input_data[index] >= 0))) );
      counts[index] = 0;
    }
  }
} 

template <typename Dtype>
__global__ void ReductionFocalLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* label, Dtype* loss,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = - input_data[index] * ( input_data[index] < 0 ) +
           log(1 + exp(input_data[index] - 2 * input_data[index] * (input_data[index] >= 0)));
      if ( label_value > 0 ) 
        counts[index] = 1;
      else
        counts[index] = 0;
    }
  }
}


template <typename Dtype>
void FocalLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.focal_loss_param().mode() ==
      FocalLossParameter_Mode_REDUCTION) {  // REDUCTION_MODE
    Dtype * mul_data = mul_.mutable_gpu_data();  // gamma * imput + beta
    caffe_gpu_set(bottom[0]->count(), Dtype(-1.0), mul_data);
    caffe_gpu_axpy(bottom[0]->count(), Dtype(2.0), bottom[1]->cpu_data(), mul_data); // (0,1)->(-1,1)  
    caffe_gpu_mul(bottom[0]->count(), bottom[0]->cpu_data(), mul_.cpu_data(), mul_data); // x_t
    caffe_gpu_scal(bottom[0]->count(), Dtype(gamma_), mul_data);
    caffe_gpu_add_scalar(bottom[0]->count(), Dtype(beta_), mul_data);
    sigmoid_bottom_vec_[0] = &mul_;
  }
  else
    sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);

  const Dtype* prob_data = sigmoid_output_->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const Dtype* input_data = (this->layer_param_.focal_loss_param().mode()
                            == FocalLossParameter_Mode_FULL)?
                            bottom[0]->gpu_data() : mul_.gpu_data();
  const int nthreads = bottom[0]->count();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = mul_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  
  switch (this->layer_param_.focal_loss_param().mode()) {
  case FocalLossParameter_Mode_FULL:  // FULL_MODE  
    FullFocalLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, input_data, prob_data, label, loss_data,
      alpha_, gamma_,has_ignore_label_, ignore_label_, counts);
      break;
  case FocalLossParameter_Mode_REDUCTION:  // FULL_MODE  
    ReductionFocalLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, input_data, label, loss_data,
      has_ignore_label_, ignore_label_, counts);
      break;
  default:
    LOG(FATAL) << "Unknown focal loss mode.";
  }

  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss);
  if (this->layer_param_.focal_loss_param().mode() ==
     FocalLossParameter_Mode_REDUCTION)
     loss /= gamma_; 
  
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  // LOG(INFO) << valid_count;
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                        valid_count);
  // Fix a bug, which happens when propagate_down[0] = false in backward
//   caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff()); 
}


template <typename Dtype>
__global__ void FullFocalLossBackwardGPU(const int nthreads, const Dtype * input_data, 
          const Dtype* prob_data,  const Dtype* label, Dtype* bottom_diff, 
          const bool has_ignore_label_, const int ignore_label_, 
          const float gamma, const float alpha, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      bottom_diff[index] = 0;
      counts[index] = 0;
    } else if ( label_value > 0 ) {
 //     bottom_diff[index] = ( input_data[index] * (input_data[index] < 0) - log(1 + exp(input_data[index]
 //                          - 2 * input_data[index] * (input_data[index] >= 0))) );
      bottom_diff[index] = alpha * pow(Dtype(1) - prob_data[index], Dtype(gamma)) *  
                           ( gamma * prob_data[index] * log( prob_data[index] + kLOG_THRESHOLD ) + prob_data[index] - 1 );
      counts[index] = 1;
    }
    else {
  //    bottom_diff[index] = (- input_data[index] * (input_data[index] > 0)
  //                          - log(1 + exp(input_data[index] - 2 * input_data[index] * (input_data[index] >= 0))) );
      bottom_diff[index] = - (1-alpha) * pow( prob_data[index], Dtype(gamma) ) * 
                           ( gamma * (1 - prob_data[index]) * log( 1 - prob_data[index] + kLOG_THRESHOLD ) - prob_data[index] );
      counts[index] = 0; 
    } 
  }
}

template <typename Dtype>
__global__ void ReductionFocalLossBackwardGPU(const int nthreads, const Dtype* prob_data,
          const Dtype* label, Dtype* bottom_diff, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      bottom_diff[index] = 0;
      counts[index] = 0;
    } else if ( label_value == 0 ) {
      bottom_diff[index] = 1 - prob_data[index];
      counts[index] = 0;
    } else {
      bottom_diff[index] = prob_data[index] - 1;
      counts[index] = 1;
    } 
  }
}


template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int nthreads = bottom[0]->count();
    const Dtype* input_data = bottom[0]->gpu_data();
    const Dtype* prob_data = sigmoid_output_->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* counts = mul_.mutable_gpu_diff();

    switch (this->layer_param_.focal_loss_param().mode()) {
    case FocalLossParameter_Mode_FULL:  // FULL_MODE
      FullFocalLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, input_data, prob_data, label, bottom_diff,
        has_ignore_label_, ignore_label_, gamma_, alpha_,counts);
      break;
    case FocalLossParameter_Mode_REDUCTION: // REDUCTION_MODE
      ReductionFocalLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads,  prob_data, label, bottom_diff,
        has_ignore_label_, ignore_label_, counts);
    //  caffe_copy(count, sigmoid_output_data, bottom_diff);
    //  caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown focal loss mode.";
    }
 
     Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_,
                                                        valid_count);;
    caffe_gpu_scal(nthreads, loss_weight, bottom_diff);
  }
  else
    caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff()); 
}

INSTANTIATE_LAYER_GPU_FUNCS(FocalLossLayer);


}  // namespace caffe
