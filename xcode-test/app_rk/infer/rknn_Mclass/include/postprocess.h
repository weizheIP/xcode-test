#ifndef _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
#define _RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_

#include <stdint.h>

#include <iostream>
#include <vector>

#define OBJ_NAME_MAX_SIZE 16
#define OBJ_NUMB_MAX_SIZE 2000
#define OBJ_CLASS_NUM     3

#define NMS_THRESH        0.6
// #define NMS_THRESH        0.7 //鸡笼


#define BOX_THRESH        0.1
#define PROP_BOX_SIZE     (5+OBJ_CLASS_NUM)

typedef struct _BOX_RECT
{
    int left;
    int right;
    int top;
    int bottom;
} BOX_RECT;

typedef struct __detect_result_t
{
    char name[OBJ_NAME_MAX_SIZE];
    BOX_RECT box;
    float prop;
} detect_result_t;

typedef struct _detect_result_group_t
{
    int id;
    int count;
    detect_result_t results[OBJ_NUMB_MAX_SIZE];
} detect_result_group_t;

int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);


int post_process(int8_t *input0, int8_t *input1, int8_t *input2,int8_t *input3, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

int post_process(int8_t *input0, int8_t *input1, int8_t *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold,int class_num, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);


//3层
int post_process(float *input0, float *input1, float *input2, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold,int class_num, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

//4层
int post_process(float *input0, float *input1, float *input2,float *input3, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold,int class_num, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);


int post_process(int8_t *input0, int8_t *input1, int8_t *input2,int8_t *input3, int model_in_h, int model_in_w,
                 float conf_threshold, float nms_threshold,int class_num, float scale_w, float scale_h,
                 std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                 detect_result_group_t *group);

int post_process(float* input0, float* input1, float* input2, float* input3, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, int class_num,float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales, detect_result_group_t* group);

std::vector<float> post_process_v8(int8_t *input0, int model_in_h, int model_in_w,
                    float conf_threshold, float nms_threshold, int class_num, float scale_w, float scale_h,
                    std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                    detect_result_group_t *group);


std::vector<float> post_process_v8(float *input0, int model_in_h, int model_in_w,
                    float conf_threshold, float nms_threshold, int class_num, float scale_w, float scale_h,
                    std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                    detect_result_group_t *group);


std::vector<float> post_process_v8(float *input0, int model_in_h, int model_in_w, int totle_strides,
                    float conf_threshold, float nms_threshold, int class_num, float scale_w, float scale_h,
                    std::vector<int32_t> &qnt_zps, std::vector<float> &qnt_scales,
                    detect_result_group_t *group);





void deinitPostProcess();


#endif //_RKNN_ZERO_COPY_DEMO_POSTPROCESS_H_
