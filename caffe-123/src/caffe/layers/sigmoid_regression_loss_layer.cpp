#include <vector>

#include "caffe/layers/sigmoid_regression_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidRegressionLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
  
//  beta_ = this->layer_param_.sigmoid_regression_loss_param().beta();
//  LOG(INFO) << beta_;
}

template <typename Dtype>
void SigmoidRegressionLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  num_ = bottom[0]->num();
  count_ = bottom[0]->count();
}

template <typename Dtype>
Dtype SigmoidRegressionLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(count_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
      //  normalizer = Dtype(count_);
        normalizer = Dtype(num_);
      } else {
        // Dtype dim_ = count_/num_;
        // normalizer = Dtype(valid_count/dim_);
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
} 


template <typename Dtype>
void SigmoidRegressionLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
//  const int num = bottom[0]->num();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;
  int valid_count = 0;
  for (int i = 0; i < count; ++i) {
    const int label_value = static_cast<int>(target[i]);
    if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
    }  
    loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    valid_count++;
   /*  loss -= ( input_data[i] * target[i] * (input_data[i] < 0) 
          + beta_ * input_data[i] * (target[i]-1.0) * (input_data[i] >= 0)
          - (target[i]+beta_*(1-target[i]))*log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)))); */
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, valid_count);
}

template <typename Dtype>
void SigmoidRegressionLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
 //   const int num = bottom[0]->num();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int valid_count = 0;
    for (int i = 0; i < count; ++i) {
      const int label_value = static_cast<int>(target[i]);
      if (has_ignore_label_ && label_value == ignore_label_) {
          bottom_diff[i] = 0;
      }
      else {
          bottom_diff[i] = sigmoid_output_data[i] - target[i];
          ++valid_count;
      }
    }
    // sigmoid cross entropy backward propagation  
    // caffe_sub(count, sigmoid_output_data, target, bottom_diff);
    
    /* caffe_mul(count, target, sigmoid_output_data, bottom_diff);
    caffe_scal(count, Dtype(1.0-beta_), bottom_diff);
    caffe_axpy(count, Dtype(beta_), sigmoid_output_data, bottom_diff);
    caffe_axpy(count, Dtype(-1.0), target, bottom_diff); */
    
    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, valid_count);
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

//#ifdef CPU_ONLY
//STUB_GPU(SigmoidRegressionLossLayer);
//#endif

INSTANTIATE_CLASS(SigmoidRegressionLossLayer);
REGISTER_LAYER_CLASS(SigmoidRegressionLoss);

}  // namespace caffe
