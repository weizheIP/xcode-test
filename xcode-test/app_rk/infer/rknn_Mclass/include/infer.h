#pragma once

#include "common/opencv_header.h"

// #include "opencv2/core/core.hpp"
// #include "opencv2/imgproc.hpp"
// #include "opencv2/imgcodecs.hpp"
// #include "opencv2/video.hpp"
// #include "opencv2/videoio.hpp"

// #include "im2d.h"
// #include "RgaUtils.h"
// #include "rga.h"
#include "rknn_api.h"
#include "./postprocess.h"
// #ifdef MPP_DEC
// #include "prefer/MPP/Codec.h"
// #endif

#include "post/base.h"


struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;	 // Non-maximum suppression threshold
	float objThreshold;	 // Object Confidence threshold
};
// typedef struct BoxInfo
// {
// 	float x1 = 0.;
// 	float y1 = 0.;
// 	float x2 = 0.;
// 	float y2 = 0.; 
// 	float score = 0.;
// 	int label = 0;
// } BoxInfo;

// int rknn_main_detect(cv::Mat& orig_img);

class RKNN_Detector{
private:
    const char *model_name;
    rknn_context ctx;
    void *resize_buf;
    rknn_input_output_num io_num;
	int mModelH,mModelW,nChannel;

	std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
	int outPutNum_;
public:
    cv::Mat resize_image(cv::Mat &srcimg, int *newh, int *neww, int *top, int *left, int inpHeight, int inpWidth);
    RKNN_Detector(const char* model_name_);
    int init(int coreIndex = 0);

	
    // int detect(cv::Mat &orig_img,std::vector<BoxInfo> &bbox);

	int detect(cv::Mat &orig_img,std::vector<BoxInfo>& bbox,int class_num);
	int detect_v8(cv::Mat &orig_img,std::vector<BoxInfo>& bbox,int class_num);
	
	// #ifdef MPP_DEC
	// cv::Mat resize_image2(MppDecMocBuf &mocBuf, int *newh, int *neww, int *top, int *left,int inpHeight, int inpWidth);

	// int detect(MppDecMocBuf &mocBuf,std::vector<BoxInfo>& bbox,int class_num);
	// int detect_v8(MppDecMocBuf &mocBuf,std::vector<BoxInfo>& bbox,int class_num);
	// #endif

	~RKNN_Detector();
};
