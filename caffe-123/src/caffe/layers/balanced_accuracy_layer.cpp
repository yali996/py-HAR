#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/balanced_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BalancedAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  has_ignore_label_ =
    this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
  }
}

template <typename Dtype>
void BalancedAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "bottom[0], bottom[1] must have the same shape.";
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void BalancedAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype fg_acc = 0, bg_acc = 0;
  Dtype fg_count = 0, bg_count = 0;
  
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
      const int label_value = static_cast<int>(bottom_label[i]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      if (label_value) {
        fg_count ++;
        fg_acc += (bottom_data[i] >= 0 );
      } else {
        bg_count ++; 
        bg_acc += (bottom_data[i] < 0 );
      }
  }

  fg_acc /= fg_count;
  bg_acc /= bg_count;
  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = 0.5 * (fg_acc + bg_acc);
}

INSTANTIATE_CLASS(BalancedAccuracyLayer);
REGISTER_LAYER_CLASS(BalancedAccuracy);

}  // namespace caffe
