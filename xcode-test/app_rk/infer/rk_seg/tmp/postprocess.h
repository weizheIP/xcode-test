#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#include <stdint.h>
#include <vector>
#include <iostream>
#include "opencv2/opencv.hpp"

// #define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 200//64
#define OBJ_CLASS_NUM     1
#define NMS_THRESH        0.6 //0.5要不猪的会多框
#define BOX_THRESH        0.1
// #define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM+32)

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    // char name[OBJ_NAME_MAX_SIZE];
    int name;
    BOX_RECT box;
    float prop;
    int  area;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int post_process(float *input0, float *input1, float *input2, float *input3, float *maskData, int model_in_h, int model_in_w,
// int post_process(float *input0, float *input1, float *input2, float *input3, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

// void deinitPostProcess();
#endif //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
