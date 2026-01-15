#ifndef FACE_FUN_H
#define FACE_FUN_H
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core.hpp>
#include <iostream>

#include "im2d.h"
#include "RgaUtils.h"
#include "rga.h"
#include "rknn_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dlfcn.h>

#include <stdint.h>


// typedef struct PointInfo
// {
// 	cv::Point pt;
// 	float score;
// } PointInfo;

// double __get_us(struct timeval t);
// void dump_tensor_attr(rknn_tensor_attr *attr);


class FaceRecognizer{
private:
    char *model_name;
    rknn_context ctx;
    rknn_input_output_num io_num;
	// int g_model_w=1280,g_model_h=1280;
public:
	FaceRecognizer(char* model_name_):model_name(model_name_){;}
	~FaceRecognizer();
	int init();
	std::vector<float>  feature(cv::Mat orig_img);
};



#endif

