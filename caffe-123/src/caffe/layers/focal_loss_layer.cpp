#include <vector>
#include <iostream>
#include <fstream>
#include "caffe/layers/focal_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void FocalLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  
  gamma_ = this->layer_param_.focal_loss_param().gamma();
  alpha_ = this->layer_param_.focal_loss_param().alpha();
  beta_ = this->layer_param_.focal_loss_param().beta();

  sigmoid_bottom_vec_.clear();
  if (this->layer_param_.focal_loss_param().mode() ==
      FocalLossParameter_Mode_REDUCTION)  // REDUCTION_MODE
    sigmoid_bottom_vec_.push_back(&mul_);
  else    // FULL_MODE 
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
 
  LOG(INFO) << "alpha: " << alpha_;
  LOG(INFO) << "gamma: " << gamma_;
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "FOCAL_LOSS layer inputs must have the same count.";
//  prob_.ReshapeLike(*bottom[0]);   
  mul_.ReshapeLike(*bottom[0]);
  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);

  if (bottom.size() > 2 ){
    if(bottom[2]->count()%5 == 0 )
      gt_num_ = bottom[2]->num();
    else{
      gt_num_ = 0;
      for (int i = 0; i < bottom[2]->num(); ++i)
        gt_num_ += static_cast<int>( *(bottom[2]->cpu_data()+bottom[2]->offset(i,3)));
    }
  }else
     gt_num_ = 1;
}

template <typename Dtype>
Dtype FocalLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(count_);
      break;
    case LossParameter_NormalizationMode_VALID:
        normalizer = Dtype(std::max(valid_count,100)); 
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
void FocalLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  if (this->layer_param_.focal_loss_param().mode() == 
      FocalLossParameter_Mode_REDUCTION) {  // REDUCTION_MODE
    Dtype * mul_data = mul_.mutable_cpu_data();  // gamma * imput + beta
    caffe_set(bottom[0]->count(), Dtype(-1.0), mul_data);
    caffe_axpy(bottom[0]->count(), Dtype(2.0), bottom[1]->cpu_data(), mul_data); // (0,1)->(-1,1)  
    caffe_mul(bottom[0]->count(), bottom[0]->cpu_data(), mul_.cpu_data(), mul_data); // x_t
    caffe_scal(bottom[0]->count(), Dtype(gamma_), mul_data);
    caffe_add_scalar(bottom[0]->count(), Dtype(beta_), mul_data);
    sigmoid_bottom_vec_[0] = &mul_;
  }   
  else
    sigmoid_bottom_vec_[0] = bottom[0];
 
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  count_ = bottom[0]->count();
  // Stable version of loss computation from input data
  const Dtype* prob_data = sigmoid_output_->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* input_data = (this->layer_param_.focal_loss_param().mode() 
                            == FocalLossParameter_Mode_FULL)?
                            bottom[0]->cpu_data() : mul_.cpu_data();

  Dtype loss = 0;
  int valid_count = 0;
  switch (this->layer_param_.focal_loss_param().mode()) {
  case FocalLossParameter_Mode_FULL:  // FULL_MODE
    for (int i = 0; i < count_; ++i) {
      if (has_ignore_label_ && int(target[i]) == ignore_label_)
        continue;
      if ( target[i] > 0 ){
        loss -= alpha_ * pow(1-prob_data[i], Dtype(gamma_)) * 
           ( input_data[i] * (input_data[i] < 0) - log(1 + exp(input_data[i] 
             - 2 * input_data[i] * (input_data[i] >= 0))) );
         valid_count++;
      }
      else if ( target[i] == 0 )
        loss -= (1.0 - alpha_ ) * pow(prob_data[i], Dtype(gamma_)) * 
           ( - input_data[i] * (input_data[i] >= 0) 
             - log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))) );
       //  valid_count++;
    }
    break;
  case FocalLossParameter_Mode_REDUCTION:  // REDUCTION_MODE 
    // const Dtype* mul_data = mul_.cpu_data();
    for (int i = 0; i < count_; ++i) {
      if (has_ignore_label_ && int(target[i]) == ignore_label_)
        continue;
      loss += - input_data[i] * ( input_data[i] < 0 ) + 
           log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
      if (target[i] > 0) 
        valid_count++;
    } 
    loss /= gamma_;
    break;
  default:
    LOG(FATAL) << "Unknown focal loss mode.";
  }
//  LOG(INFO) << loss << " " << valid_count;
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, valid_count);
}

template <typename Dtype>
void FocalLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count_ = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int valid_count = 0;
   
    caffe_set( count_, Dtype(0), bottom_diff );  
    switch (this->layer_param_.focal_loss_param().mode()) {
    case FocalLossParameter_Mode_FULL:  // FULL_MODE 
      for ( int i=0; i < count_; i++){
        const int label_value = static_cast<int>(target[i]);
        if (has_ignore_label_ && label_value == ignore_label_)
          continue;
        const Dtype p_value = sigmoid_output_data[i];
        if (label_value > 0 ){
          bottom_diff[i] = alpha_ * pow( Dtype(1.0)-p_value, Dtype(gamma_)) 
                         * ( gamma_ * p_value * log( p_value + kLOG_THRESHOLD ) + p_value - 1 );
          valid_count++; 
        }         
        else
          bottom_diff[i] = - (1-alpha_) * pow( p_value, Dtype(gamma_)) 
                           * ( gamma_ * ( 1 - p_value ) * log( 1 - p_value + kLOG_THRESHOLD ) - p_value );
      //  valid_count++;          
      }
      break;     
    case FocalLossParameter_Mode_REDUCTION: // REDUCTION_MODE
      for ( int i=0; i < count_; i++){
        const int label_value = static_cast<int>(target[i]);
        if (has_ignore_label_ && label_value == ignore_label_)
          continue;
        if (label_value > 0 )
          bottom_diff[i] = sigmoid_output_data[i] - 1.0;
        else
          bottom_diff[i] = 1.0 - sigmoid_output_data[i];
        if (label_value > 0) 
           valid_count++; 
      }
      break;
    default:
      LOG(FATAL) << "Unknown focal loss mode.";
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, valid_count);
    caffe_scal(count_, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(FocalLossLayer);
#endif 

INSTANTIATE_CLASS(FocalLossLayer);
REGISTER_LAYER_CLASS(FocalLoss);

}  // namespace caffe
