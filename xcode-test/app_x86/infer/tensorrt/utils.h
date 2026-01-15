#ifndef __CV_UTILS
#define __CV_UTILS
#include <algorithm> 
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <cmath>
#include <numeric> // std::iota 

// using  namespace cv;

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)
// struct alignas(float) Detection {
// 	//center_x center_y w h
// 	float bbox[4];
// 	float conf;  // bbox_conf * cls_conf
// 	int class_id;
// };
static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h, std::vector<int>& padsize) {
	int w, h, x, y;
	float r_w = input_w / (img.cols*1.0);
	float r_h = input_h / (img.rows*1.0);
	if (r_h > r_w) {//宽大于高
		w = input_w;
		h = r_w * img.rows;
		x = 0;
		y = (input_h - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = input_h;
		x = (input_w - w) / 2;
		y = 0;
	}
	cv::Mat re(h, w, CV_8UC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
	cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
	padsize.push_back(h);
	padsize.push_back(w);
	padsize.push_back(y);
	padsize.push_back(x);// int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
	// cv::cvtColor(out, out, cv::COLOR_BGR2RGB); //后面转了
	return out;
}

#endif
