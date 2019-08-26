#include <iostream>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <vector>
#include <utility>
#include <functional>

#include "caffe/common.hpp"
#include "caffe/util/rpn.hpp"
#include "caffe/util/rng.hpp"

using namespace std;
using std::max;
using std::min;
using std::floor;
using std::ceil;
using std::round;

/* #define MAX(a,b) a>=b?a:b
 #define MIN(a,b) a<b?a:b */

namespace caffe {

template<typename Dtype>
void _whctrs(const vector<Dtype> anchor, vector<Dtype> & whxy)
//Return width, height, x center, and y center for an anchor(window).
{
	whxy[0] = anchor[2] - anchor[0] + 1;
	whxy[1] = anchor[3] - anchor[1] + 1;
	whxy[2] = anchor[0] + (whxy[0] - 1)/2;
	whxy[3] = anchor[1] + (whxy[1] - 1)/2;
	return;                     
}

template void _whctrs(const vector<float> anchor, vector<float> & whxy);
template void _whctrs(const vector<double> anchor, vector<double> & whxy);


template<typename Dtype>
void _mkanchors(const vector<Dtype> mwhxy, vector<Dtype> & mkanchors)
//Given a vector of widths(ws) and heights(hs) around a center
//(x_ctr, y_ctr), output a set of anchors(windows).
{
	mkanchors[0] = mwhxy[2] - (mwhxy[0] - 1)/2;
	mkanchors[1] = mwhxy[3] - (mwhxy[1] - 1)/2;
	mkanchors[2] = mwhxy[2] + (mwhxy[0] - 1)/2;
	mkanchors[3] = mwhxy[3] + (mwhxy[1] - 1)/2;
	return;            
}

template void _mkanchors(const vector<float> mwhxy, vector<float> & mkanchors);
template void _mkanchors(const vector<double> mwhxy, vector<double> & mkanchors);


template<typename Dtype>
void _ratio_enum(const vector<Dtype> anchor, Dtype ratio, vector<Dtype> & ratio_anchor)
{
	//Enumerate a set of anchors for each aspect ratio wrt an anchor.
	vector<Dtype> whxy(4,0);
	_whctrs(anchor, whxy);
	Dtype size = whxy[0] * whxy[1];
	vector<Dtype> whxyr(4,0);
	// whxyr[0] = floor(sqrt(ratio * size)+1/2);
	// whxyr[1] = floor(sqrt(size / ratio)+1/2);
	whxyr[0] = round(sqrt(size / ratio));
        // std::cout << whxyr[0] << " " << ratio * whxyr[0] << std::endl;
	whxyr[1] = round(ratio * whxyr[0]);
	// whxyr[1] = floor(ratio * whxyr[0]);
	whxyr[2] = whxy[2];
	whxyr[3] = whxy[3];
        // std::cout << anchor[0] << " " << anchor[1] << " " << anchor[2] << " " << anchor[3] << std::endl;
        // std::cout << whxyr[0] << " " << whxyr[1] << " " << whxyr[2] << " " << whxyr[3] << std::endl;
	_mkanchors(whxyr, ratio_anchor);    
        // std::cout << ratio_anchor[0] << " " << ratio_anchor[1] << " " 
        //          << ratio_anchor[2] << " " << ratio_anchor[3] << std::endl;
	return;               
}

template void _ratio_enum(const vector<float> anchor, float ratio, vector<float> & ratio_anchor);
template void _ratio_enum(const vector<double> anchor, double ratio, vector<double> & ratio_anchor);


template<typename Dtype>
void _scale_enum(const vector<Dtype> anchor, Dtype scale, vector<Dtype> & scale_anchor)
{
	//Enumerate a set of anchors for each scale wrt an anchor.
//	float *scale_anchor, *whxy;
	vector<Dtype> whxy(4,0);
	_whctrs(anchor, whxy);

	vector<Dtype> whxys(4,0);
	whxys[0] = whxy[0] * scale;
	whxys[1] = whxy[1] * scale;
	whxys[2] = whxy[2];
	whxys[3] = whxy[3];
	_mkanchors(whxys, scale_anchor);
	return;                
}

template void _scale_enum(const vector<float> anchor, float scale, vector<float> & scale_anchor);
template void _scale_enum(const vector<double> anchor, double scale, vector<double> & scale_anchor);

template<typename Dtype>
void rpn_generate_default_anchors_ss( int scale, vector<Dtype> &anchors)
{
  Dtype A[5][12] = { -15,-4,30,19,-8,-8,23,23,-3,-14,18,29,
                    -38,-16,53,31,-24,-24,39,39,-14,-36,29,51,
                    -84,-40,99,55,-56,-56,71,71,-36,-80,51,95,
                   -176,-88,191,103,-120,-120,135,135, -80,-168,95,183,
                   -360,-184,375,199,-248,-248,263,263,-168,-344,183,359};

  anchors.resize(12);
  switch(scale){
      case 2: memcpy(&anchors[0], &A[0][0], sizeof(Dtype)*anchors.size()); break;
      case 4: memcpy(&anchors[0], &A[1][0], sizeof(Dtype)*anchors.size()); break;
      case 8: memcpy(&anchors[0], &A[2][0], sizeof(Dtype)*anchors.size()); break;
      case 16: memcpy(&anchors[0], &A[3][0], sizeof(Dtype)*anchors.size()); break;
      case 32: memcpy(&anchors[0], &A[4][0], sizeof(Dtype)*anchors.size()); break;
      default: memcpy(&anchors[0], A+24, sizeof(Dtype)*anchors.size()); break;
  }
  return;
}

template void rpn_generate_default_anchors_ss( int scale, vector<float> &anchors);
template void rpn_generate_default_anchors_ss( int scale, vector<double> &anchors);


template<typename Dtype>
void rpn_generate_default_anchors( vector<int> scales, vector<Dtype> &anchors)
{
   int scales_num = scales.size();
   anchors.resize(scales_num*3*4);
   vector<Dtype> anchors_ss;
   for(int i=0; i<scales.size(); i++){
      rpn_generate_default_anchors_ss(scales[i], anchors_ss);
      memcpy(&anchors[i*12], &anchors_ss[0], sizeof(Dtype)*anchors_ss.size());
   }
  return;
}

template void rpn_generate_default_anchors( vector<int> scales, vector<float> &anchors);
template void rpn_generate_default_anchors( vector<int> scales, vector<double> &anchors);

/*******************************************************************
The function is to generate default anchors
*******************************************************************/

template <typename Dtype>
void rpn_generate_anchors(const vector<Dtype> base_anchor, const vector<Dtype> ratios, 
                          const vector<Dtype> scales, vector<Dtype> &anchors)
{
    const int num_scales = scales.size();
    vector<Dtype> ratio_anchor(4,0);
    vector<Dtype> scale_anchor(4,0);
     //construct ratio_anchors
    // std::cout << ratios.size() << std::endl;
    for (int i=0; i < ratios.size(); i++){ 
      _ratio_enum( base_anchor, ratios[i], ratio_anchor);
      for(int j=0; j<num_scales; j++){
         _scale_enum(ratio_anchor, scales[j], scale_anchor);
         anchors.push_back(scale_anchor[0]);
         anchors.push_back(scale_anchor[1]);
         anchors.push_back(scale_anchor[2]);
         anchors.push_back(scale_anchor[3]);
      }
    }
    return;
}


template void rpn_generate_anchors(const vector<float> base_anchor, const vector<float> ratios,
                                   const vector<float> scales, vector<float> &anchors);
template void rpn_generate_anchors(const vector<double> base_anchor, const vector<double> ratios, 
                                   const vector<double> scales, vector<double> &anchors);

template<typename Dtype>
void generate_default_dense_anchors(const int feat_stride, vector<Dtype> &anchors)
{
     Dtype ratio_[3] = { 0.5, 1, 2 };
     Dtype base_anchor_[4] = { 0, 0, feat_stride-Dtype(1), feat_stride-Dtype(1)};
//     Dtype scales_[3] = { pow(Dtype(2), Dtype(2)), pow(Dtype(2), Dtype(7.0/3)), pow(Dtype(2),Dtype(8.0/3)) };
     Dtype scales_[2] = { pow(Dtype(2), Dtype(2)), pow(Dtype(2), Dtype(5.0/2)) };
     vector<Dtype> ratios(ratio_, ratio_+3);
     vector<Dtype> base_anchor(base_anchor_, base_anchor_+4);     
//     vector<Dtype> scales(scales_, scales_+3);     
     vector<Dtype> scales(scales_, scales_+2);     
     rpn_generate_anchors(base_anchor, ratios, scales, anchors);	
     return;
}

template void generate_default_dense_anchors( const int feat_stride, vector<float> &anchors);
template void generate_default_dense_anchors( const int feat_stride, vector<double> &anchors);

template<typename Dtype>
void rpn_generate_dense_anchors(const vector<Dtype> base_anchor, const vector<Dtype> scales, vector<Dtype> &anchors)
{
     Dtype ratio_[3] = { 0.5, 1, 2 };
     // Dtype base_anchor_[4] = { 0, 0, 15, 15};
     vector<Dtype> ratios(ratio_, ratio_+3);
     // vector<Dtype> base_anchor(base_anchor_, base_anchor_+4);     
     rpn_generate_anchors(base_anchor, ratios, scales, anchors);	
     return;
}

template void rpn_generate_dense_anchors( const vector<float> base_anchor, const vector<float> scales, vector<float> &anchors);
template void rpn_generate_dense_anchors( const vector<double> base_anchor,const vector<double> scales, vector<double> &anchors);

/*******************************************************************
The function is to enumerate all anchors
*******************************************************************/
template <typename Dtype>
void rpn_enumerate_all_anchors(const vector<Dtype> anchors, int feat_stride, int h, int w, Dtype * all_anchors )
// h, w is the size of feature map
{		
/*	vector< vector<Dtype> > shifts(w*h, vector<Dtype>(4,0));
	// enumerate the shifts
	int i, j;
	for(j=0; j<h; j++)
		for(i=0; i<w; i++){
			shifts[j*w+i][0] = i * feat_stride;
			shifts[j*w+i][1] = j * feat_stride;
			shifts[j*w+i][2] = i * feat_stride;
			shifts[j*w+i][3] = j * feat_stride;
		}
	
	int num_anchors = anchors.size()/4;
  //	int total_anchors = w*h*num_anchors;

//	vector< vector<float> > all_anchors(total_anchors, vector<float>(4,0));	
	int m, n;
	for(m=0; m<shifts.size(); m++)
	    for(n=0; n<num_anchors; n++ ){
			all_anchors.push_back(shifts[m][0] + anchors[4*n]);
			all_anchors.push_back(shifts[m][1] + anchors[4*n+1]);
			all_anchors.push_back(shifts[m][2] + anchors[4*n+2]);
			all_anchors.push_back(shifts[m][3] + anchors[4*n+3]);
		} */

    int A = anchors.size()/4;
 //   int total_anchors = w*h*A;
    for(int j=0; j<h; j++)
      for(int i=0; i<w; i++){
        for(int n=0; n<A; n++ ){
            int offset = ((j*w+i)*A+n)*4;
            all_anchors[offset] = i * feat_stride + anchors[4*n];       
            all_anchors[offset+1] = j * feat_stride + anchors[4*n+1];       
            all_anchors[offset+2] = i * feat_stride + anchors[4*n+2];       
            all_anchors[offset+3] = j * feat_stride + anchors[4*n+3];       
        }
      }
    return;
}


template void rpn_enumerate_all_anchors(const vector<float> anchors, int feat_stride, int h, int w, float* all_anchors);
template void rpn_enumerate_all_anchors(const vector<double> anchors, int feat_stride, int h, int w, double* all_anchors);


template <typename Dtype>
void rpn_enumerate_all_anchors(const Dtype* anchors, const int A, const int feat_stride, const int h, const int w, Dtype * all_anchors )
// h, w is the size of feature map
{
    for(int j=0; j<h; j++)
      for(int i=0; i<w; i++){
        for(int n=0; n<A; n++ ){
            int offset = ((j*w+i)*A+n)*4;
            all_anchors[offset] = i * feat_stride + anchors[4*n];
            all_anchors[offset+1] = j * feat_stride + anchors[4*n+1];
            all_anchors[offset+2] = i * feat_stride + anchors[4*n+2];
            all_anchors[offset+3] = j * feat_stride + anchors[4*n+3];
        }
      }
    return;
}


template void rpn_enumerate_all_anchors(const float* anchors, const int A, const int feat_stride, const int h, const int w, float* all_anchors);
template void rpn_enumerate_all_anchors(const double* anchors, const int A, const int feat_stride, const int h, const int w, double* all_anchors);


template <typename Dtype>
void rpn_filter_anchors( vector<Dtype> all_anchors, const Dtype * im_info, int allowed_border, vector<int> &inds_inside )
{
	// vector<int> inds_inside;
	// cout << im_info[0] << " " << im_info[1] << endl;
	int i;
	for(i=0; i<all_anchors.size()/4; i++ ){
		if(all_anchors[4*i]>=-allowed_border && 
		   all_anchors[4*i+1]>=-allowed_border &&
		   all_anchors[4*i+2]<=im_info[1]+allowed_border && 
		   all_anchors[4*i+3]<=im_info[0]+allowed_border)
		   inds_inside.push_back(i);
	}
	return;
}


template void rpn_filter_anchors( vector<float> all_anchors, const float * im_info, int allowed_border, vector<int> &inds_inside );
template void rpn_filter_anchors( vector<double> all_anchors, const double * im_info, int allowed_border, vector<int> &inds_inside );


template <typename Dtype>
Dtype bbox_overlap( Dtype xmin1, Dtype ymin1, Dtype xmax1, Dtype ymax1,
                    Dtype xmin2, Dtype ymin2, Dtype xmax2, Dtype ymax2 )
{
    Dtype x1, y1, x2, y2;
    if( xmax2 < xmin1 || ymax2 < ymin1 || xmax1 < xmin2 || ymax1 < ymin2 )
        return 0.0;
    
    x1 = max(xmin1, xmin2); 
    y1 = max(ymin1, ymin2); 
    x2 = min(xmax1, xmax2); 
    y2 = min(ymax1, ymax2);
    return ( x2-x1+1 )*(y2-y1+1)/( (xmax1-xmin1+1)*(ymax1-ymin1+1)+(xmax2-xmin2+1)*(ymax2-ymin2+1) - ( x2-x1+1 )*(y2-y1+1) );
} 

template float bbox_overlap( float xmin1, float ymin1, float xmax1, float ymax1,
                    float xmin2, float ymin2, float xmax2, float ymax2 );
template double bbox_overlap( double xmin1, double ymin1, double xmax1, double ymax1,
                    double xmin2, double ymin2, double xmax2, double ymax2 );


template<typename Dtype>
Dtype euclidean_feature_dist(const Dtype * ftrs_a, const Dtype * ftrs_b, int dim ){
   Dtype f_dist = 0.0;
   for( int i = 0; i < dim; i++ )
	   f_dist += (ftrs_b[i] - ftrs_a[i]) * (ftrs_b[i] - ftrs_a[i]);
   f_dist /= dim;
   f_dist = sqrt(f_dist);
   return f_dist;
}

template float euclidean_feature_dist(const float * ftrs_a, const float * ftrs_b, int dim );
template double euclidean_feature_dist(const double * ftrs_a, const double * ftrs_b, int dim );

/*
void mean_and_stds(const vector<float> targets, int M, int N, vector<float> & mean, vector<float> & stds)
{
	mean.clear();
	stds.clear();

	int m, n;
	for(n=0; n<N; n++){
		mean[n] = 0.0;
		stds[n] = 0.0;
	}

	for(m=0; m<M; m++)
		for(n=0; n<N; n++){
			mean[n] += targets[m*N+n]/M; 
			stds[n] += targets[m*N+n]*targets[m*N+n]/M;
		}

	for(n=0; n<N; n++){
		stds[n] -= mean[n] * mean[n];
		stds[n] = sqrt(stds[n]);
	}
	return;
} */

template<typename Dtype>
void rpn_normalize_targets( vector<Dtype> &targets, vector<Dtype> mean, vector<Dtype> stds )
{
       const int N = mean.size();
       const int M = targets.size()/N;
	int m, n;
	for(m=0; m<M; m++)
	    for(n=0; n<N; n++){
		targets[m*N+n] -= mean[n];
		targets[m*N+n] /= stds[n];
	    }
	return;
} 

template void rpn_normalize_targets( vector<float> &targets, vector<float> mean, vector<float> stds );
template void rpn_normalize_targets( vector<double> &targets, vector<double> mean, vector<double> stds );

template<typename Dtype>
void normalize_targets( Dtype* targets, const int M, const vector<Dtype> mean, const vector<Dtype> stds )
{
       const int N = mean.size();
       // const int M = targets.size()/N;
	int m, n;
	for(m=0; m<M; m++)
	    for(n=0; n<N; n++){
		targets[m*N+n] -= mean[n];
		targets[m*N+n] /= stds[n];
	    }
	return;
} 

template void normalize_targets( float* targets, const int M, const vector<float> mean, const vector<float> stds );
template void normalize_targets( double* targets,const int M, const vector<double> mean, const vector<double> stds );


template<typename Dtype>
void rpn_unnormalize_targets( vector<Dtype> &targets, vector<Dtype> mean, vector<Dtype> stds )
{
       const int N = mean.size();
       const int M = targets.size()/N;
	int m, n;
	for(m=0; m<M; m++)
	    for(n=0; n<N; n++){
		targets[m*N+n] *= stds[n];
		targets[m*N+n] += mean[n];
	    }
	return;
} 

template void rpn_unnormalize_targets( vector<float> &targets, vector<float> mean, vector<float> stds );
template void rpn_unnormalize_targets( vector<double> &targets, vector<double> mean, vector<double> stds );


template <typename Dtype>
void rpn_get_regression_labels(const vector <Dtype> anchors, int num_clss, vector<Dtype> &bbox_targets,
						   vector<Dtype> & bbox_inside_weights )
{
	int start;
	int num_rois = anchors.size();
	for(int n=0; n<num_rois; n++)
		for(int c=0; c<num_clss; c++){
			start = (n*num_rois+c)*4;
			if(c==anchors[n*5]){
				bbox_targets[start] = anchors[n*5+1];
				bbox_targets[start+1] = anchors[n*4+2];
				bbox_targets[start+2] = anchors[n*4+3];
				bbox_targets[start+3] = anchors[n*4+4];

				bbox_inside_weights[start] = 0.1;
				bbox_inside_weights[start+1] = 0.1;
				bbox_inside_weights[start+2] = 0.1;
				bbox_inside_weights[start+3] = 0.1;
			}
			else{
				bbox_targets[start] = 0.0;
				bbox_targets[start+1] = 0.0;
				bbox_targets[start+2] = 0.0;
				bbox_targets[start+3] = 0.0;

				bbox_inside_weights[start] = 0.0;
				bbox_inside_weights[start+1] = 0.0;
				bbox_inside_weights[start+2] = 0.0;
				bbox_inside_weights[start+3] = 0.0;
			}
		}
	return;
}


template void rpn_get_regression_labels(const vector <float> anchors, int num_clss, vector<float> &bbox_targets, vector<float> & bbox_inside_weights );
template void rpn_get_regression_labels(const vector <double> anchors, int num_clss, vector<double> &bbox_targets, vector<double> & bbox_inside_weights );

void generate_random_sequence(int max_num, int num, vector<int> & rand_inds )
{
	vector <int> vb_inds(max_num, 0);
	int ind, temp, i;
	for(i=0; i<max_num; i++)
		vb_inds[i] = i;
	
	rand_inds.clear();
	for(i=0; i<num; i++ ){
		ind = rand()%(max_num-i);
		rand_inds.push_back(vb_inds[ind]);
		temp = vb_inds[ind];
		vb_inds[ind] = vb_inds[max_num-i-1];
		vb_inds[max_num-i-1] = temp;
	}
	return;
}

template <typename Dtype>
void rpn_bbox_transform(vector<Dtype> ex_rois, vector<Dtype> gt_rois,  vector<Dtype> &bbox_targets )
{
	Dtype ex_cx, ex_cy, ex_w, ex_h;
	Dtype gt_cx, gt_cy, gt_w, gt_h;
	for(int n=0; n<ex_rois.size()/4; n++){
		ex_w = ex_rois[4*n+2] - ex_rois[4*n]+1;
		ex_h = ex_rois[4*n+3] - ex_rois[4*n+1]+1;
		ex_cx = ex_rois[4*n] + ex_w/2;
		ex_cy = ex_rois[4*n+1] + ex_h/2;

		gt_w = gt_rois[4*n+2] - gt_rois[4*n]+1;
		gt_h = gt_rois[4*n+3] - gt_rois[4*n+1]+1;
		gt_cx = gt_rois[4*n] + gt_w/2;
		gt_cy = gt_rois[4*n+1] + gt_h/2;

		bbox_targets[4*n] = (gt_cx - ex_cx)/ex_w;
		bbox_targets[4*n+1] = (gt_cy - ex_cy)/ex_h;
		bbox_targets[4*n+2] = log(gt_w/ex_w);
		bbox_targets[4*n+3] = log(gt_h/ex_h);
	}
	return;
} 

template void rpn_bbox_transform(vector<float> ex_rois, vector<float> gt_rois,  vector<float> &bbox_targets );
template void rpn_bbox_transform(vector<double> ex_rois, vector<double> gt_rois,  vector<double> &bbox_targets );


template <typename Dtype>
void bbox_transform(const Dtype * ex_rois, const Dtype * gt_rois, const int num_rois, Dtype * bbox_targets )
{
	Dtype ex_cx, ex_cy, ex_w, ex_h;
	Dtype gt_cx, gt_cy, gt_w, gt_h;
	for(int n=0; n<num_rois; n++){
		ex_w = ex_rois[4*n+2] - ex_rois[4*n]+1;
		ex_h = ex_rois[4*n+3] - ex_rois[4*n+1]+1;
		ex_cx = ex_rois[4*n] + ex_w/2;
		ex_cy = ex_rois[4*n+1] + ex_h/2;

		gt_w = gt_rois[4*n+2] - gt_rois[4*n]+1;
		gt_h = gt_rois[4*n+3] - gt_rois[4*n+1]+1;
		gt_cx = gt_rois[4*n] + gt_w/2;
		gt_cy = gt_rois[4*n+1] + gt_h/2;

		bbox_targets[4*n] = (gt_cx - ex_cx)/ex_w;
		bbox_targets[4*n+1] = (gt_cy - ex_cy)/ex_h;
		bbox_targets[4*n+2] = log(gt_w/ex_w);
		bbox_targets[4*n+3] = log(gt_h/ex_h);
	}
	return;
} 

template void bbox_transform(const float* ex_rois, const float* gt_rois, const int num_rois, float* bbox_targets );
template void bbox_transform(const double* ex_rois, const double* gt_rois, const int num_rois, double* bbox_targets );




template <typename Dtype>
void rpn_bbox_transform_inv(const Dtype * boxes, const Dtype * deltas, const int N, const int M, vector<Dtype> & pred_boxes)
{
 //	pred_boxes.resize(M*N*4);
	Dtype cx, cy, w, h;
	Dtype px, py, pw, ph;
        for(int n=0; n<N; n++){
		w = boxes[4*n+2] - boxes[4*n]+1.0;
		h = boxes[4*n+3] - boxes[4*n+1]+1.0;
		cx = boxes[4*n] + 0.5*w - 1.0;
		cy = boxes[4*n+1] + 0.5*h - 1.0;

		for(int m=0; m<M; m++){
			px = deltas[(n*M+m)*4]*w+cx;
			py = deltas[(n*M+m)*4+1]*h+cy;
			pw = exp(deltas[(n*M+m)*4+2]) * w;
			ph = exp(deltas[(n*M+m)*4+3]) * h;

			pred_boxes[(n*M+m)*4] = px - 0.5 * pw;
			pred_boxes[(n*M+m)*4+1] = py - 0.5 * ph;
			pred_boxes[(n*M+m)*4+2] = px + 0.5 * pw;
			pred_boxes[(n*M+m)*4+3] = py + 0.5 * ph;
		}
	}
	return;
}

template void rpn_bbox_transform_inv(const float * boxes, const float * deltas, 
		                             const int N, const int M, vector<float> & pred_boxes);
template void rpn_bbox_transform_inv(const double * boxes, const double * deltas, 
		                             const int N, const int M, vector<double> & pred_boxes);

	
template <typename Dtype>
void rpn_bbox_transform_inv(vector<Dtype> boxes, vector<Dtype> deltas, vector<Dtype> & pred_boxes)
{
	const int N = boxes.size()/4;
	const int M = deltas.size()/(N*4);
    
    rpn_bbox_transform_inv(&boxes[0], &deltas[0],  N,  M, pred_boxes);
    return;
}
template void rpn_bbox_transform_inv(vector<float> boxes, vector<float> deltas, vector<float> & pred_boxes);
template void rpn_bbox_transform_inv(vector<double> boxes, vector<double> deltas, vector<double> & pred_boxes);


template <typename Dtype>
void rpn_clip_boxes(vector<Dtype> & boxes, const Dtype * im_shape )
{
	int M = boxes.size()/4;
	for(int n=0; n<M; n++){
	        boxes[4*n] = min(im_shape[1]-static_cast<Dtype>(1.0),max(static_cast<Dtype>(0.0),boxes[4*n]));
		boxes[4*n+1] = min(im_shape[0]-static_cast<Dtype>(1.0),max(static_cast<Dtype>(0.0),boxes[4*n+1]));
		boxes[4*n+2] = min(im_shape[1]-static_cast<Dtype>(1.0),max(static_cast<Dtype>(0.0),boxes[4*n+2]));
		boxes[4*n+3] = min(im_shape[0]-static_cast<Dtype>(1.0),max(static_cast<Dtype>(0.0),boxes[4*n+3]));
	}
	return;
} 

template void rpn_clip_boxes(vector<float> & boxes, const float * im_shape );
template void rpn_clip_boxes(vector<double> & boxes, const double * im_shape );



template <typename Dtype>
void clip_boxes(Dtype* boxes, const int M, const Dtype * im_shape )
{
//	int M = boxes.size()/4;
	for(int n=0; n<M; n++){
	        boxes[4*n] = min(im_shape[1]-static_cast<Dtype>(1.0),max(static_cast<Dtype>(0.0),boxes[4*n]));
		boxes[4*n+1] = min(im_shape[0]-static_cast<Dtype>(1.0),max(static_cast<Dtype>(0.0),boxes[4*n+1]));
		boxes[4*n+2] = min(im_shape[1]-static_cast<Dtype>(1.0),max(static_cast<Dtype>(0.0),boxes[4*n+2]));
		boxes[4*n+3] = min(im_shape[0]-static_cast<Dtype>(1.0),max(static_cast<Dtype>(0.0),boxes[4*n+3]));
	}
	return;
} 

template void clip_boxes(float* boxes, const int M, const float * im_shape );
template void clip_boxes(double* boxes, const int M, const double * im_shape );

// This function is to filter the predicted bounding boxes with the too small size
template <typename Dtype>
void rpn_filter_small_proposals( vector<Dtype> proposals, Dtype min_size, vector<int> & keep )
{
	// vector<int> inds_inside;
	keep.clear();
	for(int i=0; i<proposals.size()/4; i++ ){
		if( proposals[4*i+2]-proposals[4*i]+1>=min_size &&
		   proposals[4*i+3]-proposals[4*i+1]+1>=min_size )
		   keep.push_back(i);
	}
	return;
}


template void rpn_filter_small_proposals( vector<float> proposals, float min_size, vector<int> & keep );
template void rpn_filter_small_proposals( vector<double> proposals, double min_size, vector<int> & keep );


template <typename Dtype>
void nms(const Dtype * proposals, const Dtype * scores, const Dtype * features,
	 const int proposals_dim, const int scores_dim, const int feature_dim, 
         const int N, const float nms_thresh, const float sim_thresh, 
         vector<int> & keep )
{
//    LOG(INFO) << proposals_dim << " " << scores_dim << " " << feature_dim << " " << N << std::endl;
    std::vector<std::pair<Dtype, int> > S(N);
    for(int i=0; i<N; i++){
        S[i] = std::make_pair(scores[i*scores_dim], i);
    }
    std::partial_sort(S.begin(), S.end(), S.end(), std::greater<std::pair<Dtype, int> >());

    int m;
    Dtype overlap, ftr_dist;
    keep.clear();
    int top;
    while(!S.empty()){
        top = S[0].second;
        keep.push_back(top);
	m=S.size()-1;
	while(m>=1){
            overlap = bbox_overlap( proposals[top*proposals_dim], proposals[top*proposals_dim+1], 
                                    proposals[top*proposals_dim+2], proposals[top*proposals_dim+3],
                                    proposals[S[m].second*proposals_dim], proposals[S[m].second*proposals_dim+1],
                                    proposals[S[m].second*proposals_dim+2], proposals[S[m].second*proposals_dim+3] );  
            
          /*  LOG(INFO) << proposals_dim << " " << scores_dim << " " << feature_dim << " " << N << std::endl;
            LOG(INFO) << proposals[top*proposals_dim] << " " << proposals[top*proposals_dim+1] << " "
                      << proposals[top*proposals_dim+2] << " " <<  proposals[top*proposals_dim+3] << " "
                      << proposals[S[m].second*proposals_dim] << " " << proposals[S[m].second*proposals_dim+1] << " "
                      << proposals[S[m].second*proposals_dim+2] << " " << proposals[S[m].second*proposals_dim+3]; */
            if(features != NULL)    
	        ftr_dist = euclidean_feature_dist( features+top*feature_dim, features+S[m].second*feature_dim, feature_dim );
            else
                ftr_dist = static_cast<Dtype>(0.0);
            // LOG(INFO) << top << " " << S[m].second << " " << ftr_dist << " " << overlap;
	    if(overlap >= nms_thresh && ftr_dist <= sim_thresh )
	    // if(overlap >= nms_thresh )
                S.erase(S.begin()+m);  m--;
	    }
	    S.erase(S.begin());  // delete the first element
    } 
    return;
}


template void nms(const float * proposals, const float * scores, const float * features,
         const int proposals_dim, const int scores_dim, const int feature_dim, 
         const int N, const float nms_thresh, const float sim_thresh, vector<int> & keep );

template void nms(const double * proposals, const double * scores, const double * features,
         const int proposals_dim, const int scores_dim, const int feature_dim, 
         const int N, const float nms_thresh, const float sim_thresh, vector<int> & keep );


template <typename Dtype>
void nms(const vector<Dtype> proposals, const vector<Dtype> scores, const vector<Dtype> features,
             const float nms_thresh, const float sim_thresh, int feature_dim, vector<int> & keep )
{
    std::vector<std::pair<Dtype, int> > S(scores.size());
    for(int i=0; i<scores.size();i++){
         S[i] = std::make_pair(scores[i], i);
    }
    std::partial_sort(S.begin(), S.end(), S.end(), std::greater<std::pair<Dtype, int> >());
    // LOG(INFO) << S.size();

    int m;
    int ftr_count = features.size();
    if(ftr_count < scores.size())
        ftr_count = 0;

    Dtype overlap, ftr_dist;
    keep.clear();
    int top;
    while(!S.empty()){
        top = S[0].second;
        keep.push_back(top);
        m=S.size()-1;
        while(m>=1){
            overlap = bbox_overlap( proposals[4*top], proposals[4*top+1],proposals[4*top+2], proposals[4*top+3],
                                    proposals[4*S[m].second], proposals[4*S[m].second+1],
                                    proposals[4*S[m].second+2], proposals[4*S[m].second+3] );
            // LOG(INFO) << overlap;
            if(ftr_count>0)
                ftr_dist = euclidean_feature_dist( &features[top*feature_dim], &features[S[m].second*feature_dim], feature_dim );
            else
                ftr_dist = static_cast<Dtype>(1.0);

            if(overlap >= nms_thresh && ftr_dist >= sim_thresh ){
                // LOG(INFO) << m << " " << S.size();
                S.erase(S.begin()+m);  
            }
            m--;
        }
        // LOG(INFO) << S.size();
        S.erase(S.begin());  // delete the first element
    }
    return;
}

template void nms(const vector<float> proposals, const vector<float> scores, const vector<float> features,
             const float nms_thresh, const float sim_thresh, int feature_dim, vector<int> & keep );
template void nms(const vector<double> proposals, const vector<double> scores, const vector<double> features,
             const float nms_thresh, const float sim_thresh, int feature_dim, vector<int> & keep );



template <typename Dtype>
void nms(const vector<Dtype> proposals, const vector<Dtype> scores, const float nms_thresh, vector<int> & keep )
{
/*   const Dtype * features = NULL;
   const float sim_thresh = 0.0;
   const int feature_dim = 0;
   const int scores_dim = 1;
   const int num_proposals = scores.size();
   const int proposals_dim = proposals.size()/num_proposals;
   
    std::cout << proposals_dim << " " << scores_dim << " " << feature_dim << " " << num_proposals << std::endl;
//   nms(proposals, scores, features, nms_thresh, sim_thresh, feature_dim, keep );
   nms(&proposals[0], &scores[0], features , proposals_dim, scores_dim, feature_dim,
       num_proposals, nms_thresh, sim_thresh, keep ); */

   const vector<Dtype> features(0);
   float sim_thresh = 0.0;
   int feature_dim = 1;
   nms(proposals, scores, features, nms_thresh, sim_thresh, feature_dim, keep );
   return;
}

template void nms(const vector<float> proposals, const vector<float> scores, const float nms_thresh, vector<int> & keep );
template void nms(const vector<double> proposals, const vector<double> scores, const float nms_thresh, vector<int> & keep );


/*template <typename Dtype>
void allocate_bags_with_nms(const vector<Dtype> proposals, const vector<Dtype> scores, const float nms_thresh, 
		           vector<int> keep, vector<int> &bag_index )
{
   int num_proposals = scores.size();
   bag_index.resize(num_proposals);
   for(int i=0; i < num_proposals; i++)
	   bag_index[i] = -1;

   for(int j=0; j<keep.size(); j++){
       for(int i=0; i<num_proposals; i++){
		 int top = keep[j];
                 Dtype ovlp = bbox_overlap( proposals[4*top], proposals[4*top+1],proposals[4*top+2], proposals[4*top+3],
	                               proposals[4*i], proposals[4*i+1], proposals[4*i+2], proposals[4*i+3] );
		 if(ovlp>= nms_thresh && bag_index[i]==-1)
			 bag_index[i] = j;
	   }
   }
  return;
}

template void allocate_bags_with_nms(const vector<float> proposals, const vector<float> scores, const float nms_thresh, 
		      vector<int> keep, vector<int> &bag_index );
template void allocate_bag_with_nms(const vector<double> proposals, const vector<double> scores, const float nms_thresh, 
		      vector<int> keep, vector<int> &bag_index );
*/


template <typename Dtype>
void get_refined_boxes(const vector<Dtype> proposals, const vector<Dtype> cls_probs, const float ovlp_thresh, 
		                  vector<int> keep, vector<Dtype> &reg_boxes )
{
  bool weight_boxes = true;
  reg_boxes.resize(keep.size()*4);
  int i, j, k;
  Dtype overlap;
  for( i=0; i<keep.size(); i++ ){
     for( j=0; j<4; j++ )
	   reg_boxes[i*4+j] = proposals[4*keep[i]+j];
     if(weight_boxes){
	   Dtype Z = cls_probs[keep[i]];
       for( j=0; j<4; j++ )
		 reg_boxes[i*4+j] *= cls_probs[keep[i]];
       for( k=0; k<cls_probs.size(); k++ ){
		   overlap = bbox_overlap( proposals[4*keep[i]], proposals[4*keep[i]+1], proposals[4*keep[i]+2], proposals[4*keep[i]+3],
				                   proposals[4*k], proposals[4*k+1],proposals[4*k+2], proposals[4*k+3] );
	       if( overlap > ovlp_thresh ){
			 Z += cls_probs[k];
			 for(j=0; j<4; j++)
		     reg_boxes[i*4+j] += proposals[k*4+j]*cls_probs[k]; 
           }
	   }
	   for(j=0; j<4; j++)
		   reg_boxes[i*4+j] /= Z;
	 }
  }
  return;
}

template void get_refined_boxes(const vector<float> proposals, const vector<float> cls_probs, const float ovlp_thresh, 
		      vector<int> keep, vector<float> &reg_boxes );
template void get_refined_boxes(const vector<double> proposals, const vector<double> cls_probs, const float ovlp_thresh, 
		      vector<int> keep, vector<double> &reg_boxes );


template <typename Dtype>
void overlap_distance(const Dtype * cls_boxes, // detected boxes
                      const Dtype * gt_boxes,  // ground truth boxes
                      const int M, // number of detections
                      const int N, // number of ground truths
                      const int dim, // dim for cls_boxes, gt_boxes
		      Dtype * D )
{
  
//  LOG(INFO) << M << " " << N << " " << dim;  
  for( int i=0; i<M; i++ )
     for( int j=0; j<N; j++ ){
         if(dim == 5 && cls_boxes[i*dim+4] != gt_boxes[j*dim+4] )
             D[i*N+j] = 1.0; 
         if(dim == 6 && ( cls_boxes[i*dim+4] != gt_boxes[j*dim+4] || cls_boxes[i*dim+5] != gt_boxes[j*dim+5] ))
             D[i*N+j] = 1.0; 
         D[i*N+j] = 1.0 - bbox_overlap( cls_boxes[i*dim], cls_boxes[i*dim+1], cls_boxes[i*dim+2], cls_boxes[i*dim+3],
   				 gt_boxes[j*dim], gt_boxes[j*dim+1], gt_boxes[j*dim+2], gt_boxes[j*dim+3] );

     }
  return;
}

template void overlap_distance(const float * cls_boxes, const float * gt_boxes, const int M, const int N, const int dim, float * D );
template void overlap_distance(const double * cls_boxes, const double * gt_boxes, const int M, const int N, const int dim, double * D );

}


