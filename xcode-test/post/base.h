#ifndef BASE
#define BASE

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <memory>
using namespace std;
// 只是检测
typedef struct BoxInfo
{
    int x1;
    int y1;
    int x2;
    int y2;
    float score;
    int label;
} BoxInfo;

// 检测和关键点
struct CarBoxInfo
{
    int x1;
    int y1;
    int x2;
    int y2;
    int label;
    float score;
    std::vector<float> landmark;
};

// 人脸关键点
typedef struct{
	int x1;
	int y1;
	int x2;
	int y2;
    // int label;
	float score;
    std::vector<cv::Point> keypoint;
	std::vector<float> feature1;
}Yolov5Face_BoxStruct;



// 检测和分割
struct Object
{
    cv::Rect rect;
    cv::Rect rect_160;
    std::vector<float> mask_feat;
    cv::Mat boxMask;
    int area; // 分割的面积
    int label;
    float prob;
    std::vector<cv::Point> cc;
};

// 只是检测
// struct Object
// {
//     // cv::Rect_<float> rect;
//     cv::Rect rect;
//     int label;
//     float prob;
// };




#endif
