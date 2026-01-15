// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <dlfcn.h>
#include "infer/rknn_Mclass/include/infer.h"
#include "common/license/license.h"
#include <algorithm>


#define _BASETSD_H

const int MaxOutPut = 10000;






static void dump_tensor_attr(rknn_tensor_attr *attr)
{
printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
        "zp=%d, scale=%f\n",
        attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
        attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
        get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    unsigned char *outdata;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);
    int nret = get_model_decrypt_value(data,outdata,sz);
    if(0 != nret)
    {
        printf("decode model failure.\n");
        return NULL;
    }
    
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{

    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }
    printf("Open model file %s .\n", filename);
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}



cv::Mat RKNN_Detector::resize_image(cv::Mat &srcimg, int *newh, int *neww, int *top, int *left,int inpHeight, int inpWidth)
{
    // int inpHeight=800;
    // int inpWidth=800;
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = inpHeight;
    *neww = inpWidth;
    cv::Mat dstimg;
    if (srch != srcw)
    {
        float hw_scale = (float)srch / srcw;
        if (hw_scale > 1)
        {
            *newh = inpHeight;
            *neww = int(inpWidth / hw_scale);
            if(*newh == srcimg.rows && *neww == srcimg.cols)
            {
                dstimg = srcimg;
            }
            else
                cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
            *left = int((inpWidth - *neww) * 0.5);
            cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, inpWidth - *neww - *left, cv::BORDER_CONSTANT, 114);
        }
        else
        {
            *newh = (int)inpHeight * hw_scale;
            *neww = inpWidth;
            if(*newh == srcimg.rows && *neww == srcimg.cols)
            {
                dstimg = srcimg;
            }
            else
                cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
            *top = (int)(inpHeight - *newh) * 0.5;
            cv::copyMakeBorder(dstimg, dstimg, *top, inpHeight - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 114);
        }
    }
    else
    {
        cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
    }
    // cv::imwrite("img_resize.jpg",dstimg);
    return dstimg;
}




RKNN_Detector::RKNN_Detector(const char* model_name_):model_name(model_name_){
    ;
}

int RKNN_Detector::init(int coreIndex){
        /* Create the neural network */
    printf("Loading mode...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_model(model_name, &model_data_size);


    int ret = rknn_init(&ctx, model_data, model_data_size, 1, NULL);
    free(model_data);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                    sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    printf("sdk version: %s driver version: %s\n", version.api_version,
        version.drv_version);

    
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input,
        io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                        sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("[INFO] rknn_init error ret=%d\n", ret);
            return -1;
        }
        // dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                        sizeof(rknn_tensor_attr));
        // dump_tensor_attr(&(output_attrs[i]));
    }
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW)
    {
        // printf("model is NCHW input fmt\n");
        nChannel = input_attrs[0].dims[1];
        mModelW = input_attrs[0].dims[2];
        mModelH = input_attrs[0].dims[3];
    }
    else
    {
        // printf("model is NHWC input fmt\n");
        mModelW = input_attrs[0].dims[1];
        mModelH = input_attrs[0].dims[2];
        nChannel = input_attrs[0].dims[3];
    }

    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

    outPutNum_ = (output_attrs[0].dims[1]/3) - 5;


    // 设置模型绑定的核心/Set the core of the model that needs to be bound
    // coreIndex=0;
    rknn_core_mask core_mask;
    switch (coreIndex)
    {
        case 1:
            core_mask = RKNN_NPU_CORE_0;
            break;
        case 2:
            core_mask = RKNN_NPU_CORE_1;
            break;
        case 3:
            core_mask = RKNN_NPU_CORE_2;
            break;
        case 5:
            core_mask = RKNN_NPU_CORE_0_1;
            break;
        case 7:
            core_mask = RKNN_NPU_CORE_0_1_2;
            break;
            
    }
    if(coreIndex)
    {
        int ret = rknn_set_core_mask(ctx, core_mask);         
    }

    return 0;
}




int RKNN_Detector::detect(cv::Mat &orig_img,std::vector<BoxInfo>& bbox,int nouse)
{
    struct timeval start_time, stop_time;
    struct timeval start_time1, stop_time1;
    struct timeval start_time2, stop_time2;
    gettimeofday(&start_time2, NULL);

    // std::vector<BoxInfo> bbox(200,0);
    if(orig_img.empty())return -1;

    // printf("[INFO] img w h: %d %d\n",orig_img.cols,orig_img.rows);

    int status = 0;
    char *model_name = NULL;
    size_t actual_size = 0;
    int img_width = 0;
    int img_height = 0;
    int img_channel = 0;
    const float nms_threshold = NMS_THRESH;
    const float box_conf_threshold = BOX_THRESH;

    int ret;

    cv::Mat img_;
    cv::cvtColor(orig_img, img_, cv::COLOR_BGR2RGB);

    if(img_.empty())
    {
        printf("[ERROR] img_ is empty\n");
        return -1;
    }
    int newh = 0, neww = 0, padh = 0, padw = 0;

    int channel = nChannel;
    int width = mModelW;
    int height = mModelH;
    

    img_width = img_.cols;
    img_height = img_.rows;

    cv::Mat img = resize_image(img_, &newh, &neww, &padh, &padw, width, height);

    

    // printf("model input height=%d, width=%d, channel=%d\n", height, width,
    //     channel);

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    // inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    inputs[0].buf = img.data;//resize_buf;
    gettimeofday(&start_time1, NULL);
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        // #ifdef FP16
        outputs[i].want_float = 1;
        // #else
        // outputs[i].want_float = 0;
        // #endif
    }
    gettimeofday(&start_time, NULL);
    ret = rknn_run(ctx, NULL);
    gettimeofday(&stop_time, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    gettimeofday(&stop_time1, NULL);
    // std::cout<<"only net time: "<<(__get_us(stop_time) - __get_us(start_time)) / 1000<<" ms, add copy: "<<(__get_us(stop_time1) - __get_us(start_time1)) / 1000<<" ms"<<std::endl;
    
    //post process
    // float scale_w = (float)width / img_width;
    // float scale_h = (float)height / img_height;

    float scale_w = 1.0f;
    float scale_h = 1.0f;


    detect_result_group_t detect_result_group;
    
    


    // #ifdef V8Detection
    //     int class_num = (output_attrs[0].dims[1]) - 16*4;
    // #else
        // int class_num = (output_attrs[0].dims[1]/3) - 5;
        int class_num = outPutNum_;
    // #endif


    #ifdef FP16
    if(io_num.n_output==4)
        post_process((float *)outputs[0].buf, (float *)outputs[1].buf, (float *)outputs[2].buf,(float*)outputs[3].buf, height, width,
                box_conf_threshold, nms_threshold,class_num, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
    #else
    if(io_num.n_output==4)
        post_process((float *)outputs[0].buf, (float *)outputs[1].buf, (float *)outputs[2].buf,(float*)outputs[3].buf, height, width,
                box_conf_threshold, nms_threshold,class_num, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
    else
        post_process((float *)outputs[0].buf, (float *)outputs[1].buf, (float *)outputs[2].buf, height, width,
            box_conf_threshold, nms_threshold,class_num, scale_w, scale_h, out_zps, out_scales, &detect_result_group);
    #endif

    // post_process((int8_t *)outputs[0].buf, (int8_t *)outputs[1].buf, (int8_t *)outputs[2].buf, height, width,
    //             box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    // Draw Objects
    float ratioh = (float)orig_img.rows / newh, ratiow = (float)orig_img.cols / neww;
    char text[256];
    
    // bbox.clear();
    int bbox_index = 0;
    for (int i = 0; i < detect_result_group.count; i++)
    {
        if(i>=MaxOutPut)break;
        //std::cout<<"i_index:"<<i<<std::endl;
        detect_result_t *det_result = &(detect_result_group.results[i]);

        int x1 = (det_result->box.left -padw)* ratiow;
        int y1 = (det_result->box.top  -padh)* ratioh;
        int x2 = (det_result->box.right -padw)* ratiow;
        int y2 = (det_result->box.bottom  -padh)* ratioh;
        

        // if((int)x1*1.<0 || (int)x2*1.<0 || (int)y1*1.<0 || (int)y2*1.<0) continue;

        x1 = std::max(std::min((float)x1, (float)(img_width - 1)), 0.f);
        y1 = std::max(std::min((float)y1, (float)(img_height - 1)), 0.f);
        
        x2 = std::max(std::min((float)x2, (float)(img_width - 1)), 0.f);
        y2 = std::max(std::min((float)y2, (float)(img_height - 1)), 0.f);
        // printf("%d,%d,%d,%d,%d,%d\n",x1,y1,x2,y2,img_.cols,img_.rows);

        BoxInfo bbox_;
        bbox_.x1 = (float)x1;
        bbox_.y1 = (float)y1;
        bbox_.x2 = (float)x2;
        bbox_.y2 = (float)y2,
        bbox_.score = (float)det_result->prop,
        bbox_.label = atoi(det_result->name);
        if(x2 -x1 <=3 || y2 -y1 <=3)
            continue;
        if(0)
        {
            std::cout<<"Label:"<<det_result->name
            <<",x1:"<<(int)x1*1.
            <<",y1:"<<(int)y1*1.
            <<",x2:"<<(int)x2*1.
            <<",y2:"<<(int)y2*1.
            <<",prop:"<<(float)det_result->prop
            <<",Name:"<<atoi(det_result->name)<<std::endl;
        }

        bbox.push_back(bbox_);
    }

    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    gettimeofday(&stop_time2, NULL);
    // std::cout<<"only net time: "<<(__get_us(stop_time) - __get_us(start_time)) / 1000<<" ms, : "<<(__get_us(stop_time1) - __get_us(start_time1)) / 1000<<" ms"<<" ms, : "<<(__get_us(stop_time2) - __get_us(start_time2)) / 1000<<" ms"<<std::endl;
    // 前后处理10ms，内存拷贝多的到20ms 

    return 0;
}



RKNN_Detector::~RKNN_Detector(){
    int ret = rknn_destroy(ctx);
    if(ret == RKNN_SUCC)
    {
        std::cout<< "RKNN release success!"<<std::endl;
    }
    else{
        std::cout<< "RKNN release false!"<<std::endl;
    }
    // delete resize_buf;
    // resize_buf = nullptr;
}




