#ifndef RK_SEG
#define RK_SEG


#include <stdint.h>
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>


#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define _BASETSD_H


#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "common/license/license.h"

// #include "rga.h"
#include "rknn_api.h"

#include "post/base.h"




// #define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 500//64
#define OBJ_CLASS_NUM     1
#define NMS_THRESH        0.6 //0.5要不猪的会多框
#define BOX_THRESH        0.1
// #define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM+32)


// yolov7-tiny
// const int anchor0[6] = {10, 13, 16, 30, 33, 23};
// const int anchor1[6] = {30, 61, 62, 45, 59, 119};
// const int anchor2[6] = {116, 90, 156, 198, 373, 326};

const int anchor0[6] = {19, 27, 44, 40, 38, 94}; // 8 
const int anchor1[6] = {96,68,86,152,180,137}; // 16
const int anchor2[6] = {140,301,303,264,238,542};// 32
const int anchor3[6] = {436,615,739,380,925,792}; // 64

typedef struct _BOX_RECT2
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT2;

// #ifdef xTDGZ
// struct BoxInfo
// {
//     int x1;
//     int y1;
//     int x2;
//     int y2;
//     float score;
//     int label;
// };
// #endif

// struct Object
// {
//     // cv::Rect_<float> rect;
//     cv::Rect rect;
//     cv::Rect rect_160;
//     std::vector<float> mask_feat;
//     cv::Mat boxMask;
//     int  area;
//     int label;
//     float prob;
//     std::vector<cv::Point> cc;
// };

typedef struct __detect_result_t2
{
    // char name[OBJ_NAME_MAX_SIZE];
    int name;
    BOX_RECT2 box;
    float prop;
    int  area;
    std::vector<cv::Point> cc;
} detect_result_t2;

typedef struct _detect_result_group_t2
{
    int id;
    int count;
    detect_result_t2 results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t2;


class rk_seg
{public:
rk_seg();
~rk_seg();


int init(std::string path);
int seg(cv::Mat &img, std::vector<Object> &segs);
int release();



private:


    int            status     = 0;
    char*          model_name = NULL;
    rknn_context   ctx;
    size_t         actual_size        = 0;
    int            img_width          = 0;
    int            img_height         = 0;
    int            img_channel        = 0;
    const float    nms_threshold      = NMS_THRESH;
    const float    box_conf_threshold = BOX_THRESH;
    struct timeval start_time, stop_time;
    int            ret;

    int channel = 3;
    int width   = 0;
    int height  = 0;

    rknn_input_output_num io_num;
    
    rknn_tensor_attr *output_attrs;

};



#endif