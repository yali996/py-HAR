#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/roi_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ROIDataLayer<Dtype>::~ROIDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ROIDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  // revised by Yali
  const int info_size = this->layer_param_.image_data_param().info_size();	
  const int roi_num = this->layer_param_.image_data_param().roi_num();	
  const int label_size = info_size + 5 * roi_num;
	
  string root_folder = this->layer_param_.image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  std::string filename;
  std::string line;
  // revised by Yali
  while (std::getline(infile, line)) {
    int label;
    std::istringstream iss(line);
    iss >> filename;
    std::vector<int> labels;
    labels.push_back(0); // if fliiped
    while( iss >> label ){
       labels.push_back(label);
    }
    lines_.push_back(std::make_pair(filename, labels));
  }
  // added by Yali Li, to add the flipped samples
  int num_samples = lines_.size();
  LOG(INFO) << num_samples;
  for (int i = 0; i < num_samples; i++){
    std::vector<int> flip_labels(lines_[i].second);
    flip_labels[0] = 1;
    // LOG(INFO) << flip_labels[info_size] << " " << flip_labels[info_size+1];
    for (int index = 0; index < flip_labels[info_size]; ++index){
      // LOG(INFO) << index;
      int temp = flip_labels[info_size+1+5*index];
      flip_labels[info_size+1+5*index] = flip_labels[1] - flip_labels[info_size+1+5*index+2] - 1;
      flip_labels[info_size+1+5*index+2] = flip_labels[1] - temp - 1;
    }
    lines_.push_back(std::make_pair(lines_[i].first, flip_labels) );
  }  
  
  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = rand_lines_ids_[0];
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = rand_lines_ids_[skip];
  }
  // Read an image, and use it to initialize the top blob.
 
  // Revised by Yali Li
  int resized_width = lines_[lines_id_].second[2];
  int resized_height = lines_[lines_id_].second[3];
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    resized_height, resized_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  vector<int> label_shape(2, batch_size);
  label_shape[1] = label_size;
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ROIDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator()); 
// revised by Yali Li
  if (rand_lines_ids_.empty() || rand_lines_ids_.size() < lines_.size() ){
    rand_lines_ids_.resize(lines_.size()); 
    for (int index = 0; index < lines_.size(); ++index)
      rand_lines_ids_[index] = index;
  }
 
  std::vector<std::pair<Dtype, int> > as_ratio(lines_.size());
  for (int index = 0; index < lines_.size(); ++index){
    as_ratio[index] = std::make_pair(static_cast<Dtype>(lines_[rand_lines_ids_[index]].second[1])
                                     /lines_[rand_lines_ids_[index]].second[2], rand_lines_ids_[index]);
  } 
  std::partial_sort(as_ratio.begin(), as_ratio.end(), as_ratio.end(), std::greater<std::pair<Dtype, int> >());
 
  std::vector<std::pair<int, int> > inds(lines_.size()/2);
  for (int index = 0; index < lines_.size()/2; ++index){
    inds[index] = std::make_pair(as_ratio[2*index].second, as_ratio[2*index+1].second);
  } 
  shuffle(inds.begin(), inds.end(), prefetch_rng);

  for (int index = 0; index < lines_.size()/2; ++index){
    rand_lines_ids_[2*index] = inds[index].first;
    rand_lines_ids_[2*index+1] = inds[index].second;
  }
  // shuffle the images 
  // shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ROIDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  const int info_size = this->layer_param_.image_data_param().info_size();	
  const int roi_num = this->layer_param_.image_data_param().roi_num();	
  const int label_size = info_size + 5 * roi_num;
  
  const int lines_size = lines_.size();
  // if (lines_id_ + batch_size > lines_size) {
  if (rand_lines_ids_.size() < batch_size ) {
    // We have reached the end. Restart from the first.
    DLOG(INFO) << "Restarting data prefetching from start.";
    lines_id_ = 0;
    if (this->layer_param_.image_data_param().shuffle()) {
      ShuffleImages();
    }
  }

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  // revisd by Yali Li
  // int resized_width = -1;
  // int resized_height = -1;
  // for (int item_id = 0; item_id < batch_size; ++item_id) {
  //  resized_width = max(resized_width, lines_[lines_id_+item_id].second[2]); 
  //  resized_height = max(resized_height, lines_[lines_id_+item_id].second[3]);
  //} 
  lines_id_ = rand_lines_ids_.back();
  
  int resized_width = lines_[lines_id_].second[3];    
  int resized_height = lines_[lines_id_].second[4];    
  
  // scale jittering
  Dtype scale_jitter = rand() / Dtype(RAND_MAX) * Dtype(0.3) + Dtype(0.7);   
  Dtype max_size_ = 1500;
  if ( resized_width * scale_jitter >= max_size_ || resized_height * scale_jitter >= max_size_ ){
     scale_jitter = max_size_ / std::max(resized_width, resized_height);
  }
  resized_width *= scale_jitter;
  resized_height *= scale_jitter;

  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      resized_height, resized_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    lines_id_ = rand_lines_ids_.back();
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        resized_height, resized_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img, static_cast<bool>(lines_[lines_id_].second[0]),
                                        &(this->transformed_data_));
    if (lines_[lines_id_].second[0] )
    trans_time += timer.MicroSeconds();

    CHECK_GT( label_size, lines_[lines_id_].second.size() ) <<
       "The input label size must be less than the prototxt setting";
    // copy the im_info
    Dtype im_scale = 0.5 * ( static_cast<Dtype>(resized_width)/lines_[lines_id_].second[1]
                  + static_cast<Dtype>(resized_height)/lines_[lines_id_].second[2]);
    prefetch_label[item_id * label_size ] = resized_height;  // height
    prefetch_label[item_id * label_size + 1] = resized_width;  // width
    prefetch_label[item_id * label_size + 2] = im_scale;                    // resize scale    
    prefetch_label[item_id * label_size + 3] = lines_[lines_id_].second[5]; // gt boxes number 
 
//    LOG(INFO) << lines_[lines_id_].second[0] << " " << lines_[lines_id_].second[1] << " "
//              << lines_[lines_id_].second[2] << " " << lines_[lines_id_].second[3] << " " 
//              << lines_[lines_id_].second[4] << " " << lines_[lines_id_].second[5];
    for( int gt_id = 0; gt_id < lines_[lines_id_].second[5]*5; ++gt_id ){ // gt boxes
      if (gt_id % 5 == 4 )
        prefetch_label[item_id*label_size +4+gt_id ] = lines_[lines_id_].second[info_size+1+gt_id];
      else
        prefetch_label[item_id*label_size +4+gt_id ] = im_scale * lines_[lines_id_].second[info_size+1+gt_id];
    }
//    for( int label_id = info_size + 1; label_id < lines_[lines_id_].second.size(); ++label_id ) // gt boxes
//       prefetch_label[item_id * label_size + label_id + 4 - (info_size+1) ] = lines_[lines_id_].second[label_id];
    // go to the next iter
    rand_lines_ids_.pop_back();
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ROIDataLayer);
REGISTER_LAYER_CLASS(ROIData);

}  // namespace caffe
#endif  // USE_OPENCV
