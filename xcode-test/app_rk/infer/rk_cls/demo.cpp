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
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rknn_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>

#include <malloc.h>
#include <iostream>


using namespace std;
using namespace cv;

/*-------------------------------------------
                  Functions
-------------------------------------------*/

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE* fp = fopen(filename, "rb");
  if (fp == nullptr) {
    printf("fopen %s fail!\n", filename);
    return NULL;
  }
  fseek(fp, 0, SEEK_END);
  int            model_len = ftell(fp);
  unsigned char* model     = (unsigned char*)malloc(model_len);
  fseek(fp, 0, SEEK_SET);
  if (model_len != fread(model, 1, model_len, fp)) {
    printf("fread %s fail!\n", filename);
    free(model);
    return NULL;
  }
  *model_size = model_len;
  if (fp) {
    fclose(fp);
  }
  return model;
}

static int rknn_GetTop(float* pfProb, float* pfMaxProb, uint32_t* pMaxClass, uint32_t outputCount, uint32_t topNum)
{
  uint32_t i, j;

#define MAX_TOP_NUM 100
  if (topNum > MAX_TOP_NUM)
    return 0;

  memset(pfMaxProb, 0, sizeof(float) * topNum);
  memset(pMaxClass, 0xff, sizeof(float) * topNum);

  for (j = 0; j < topNum; j++) {
    for (i = 0; i < outputCount; i++) {
      if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
          (i == *(pMaxClass + 4))) {
        continue;
      }

      if (pfProb[i] > *(pfMaxProb + j)) {
        *(pfMaxProb + j) = pfProb[i];
        *(pMaxClass + j) = i;
      }
    }
  }

  return 1;
}



rknn_context ctx = 0;
int            ret;
unsigned char* model;
rknn_input_output_num io_num;
const int MODEL_IN_WIDTH    = 224;
const int MODEL_IN_HEIGHT   = 224;

/*init*/
int init(const char* model_path)
{
  // const int MODEL_IN_WIDTH    = 224;
  // const int MODEL_IN_HEIGHT   = 224;
  const int MODEL_IN_CHANNELS = 3;

  // rknn_context ctx = 0;
  // int            ret;
  int            model_len = 0;
  // unsigned char* model;

  // Load RKNN Model
  model = load_model(model_path, &model_len);
  ret   = rknn_init(&ctx, model, model_len, 0, NULL);
  if (ret < 0) {
    printf("rknn_init fail! ret=%d\n", ret);
    return -1;
  }

  // Get Model Input Output Info
  // rknn_input_output_num io_num;
  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret != RKNN_SUCC) {
    printf("rknn_query fail! ret=%d\n", ret);
    return -1;
  }

  return 0;
}

int detect(cv::Mat orig_img){
  // Load image
  // cv::Mat orig_img = imread(img_path, cv::IMREAD_COLOR);
  if (!orig_img.data) {
    printf("cv::imread %s fail!\n", img_path);
    return -1;
  }

  cv::Mat orig_img_rgb;
  cv::cvtColor(orig_img, orig_img_rgb, cv::COLOR_BGR2RGB);

  cv::Mat img = orig_img_rgb.clone();
  if (orig_img.cols != MODEL_IN_WIDTH || orig_img.rows != MODEL_IN_HEIGHT) {
    printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
    // cv::resize(orig_img, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), 0, 0, cv::INTER_LINEAR);
    cv::resize(orig_img_rgb, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT));
  }

  // Set Input Data
  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index = 0;
  inputs[0].type  = RKNN_TENSOR_UINT8;
  inputs[0].size  = img.cols * img.rows * img.channels() * sizeof(uint8_t);
  inputs[0].fmt   = RKNN_TENSOR_NHWC;
  inputs[0].buf   = img.data;

  ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
  if (ret < 0) {
    printf("rknn_input_set fail! ret=%d\n", ret);
    return -1;
  }

  // Run
  printf("rknn_run\n");
  ret = rknn_run(ctx, nullptr);
  if (ret < 0) {
    printf("rknn_run fail! ret=%d\n", ret);
    return -1;
  }

  // Get Output
  rknn_output outputs[1];
  memset(outputs, 0, sizeof(outputs));
  outputs[0].want_float = 1;
  ret                   = rknn_outputs_get(ctx, 1, outputs, NULL);
  if (ret < 0) {
    printf("rknn_outputs_get fail! ret=%d\n", ret);
    return -1;
  }

  // Post Process
  for (int i = 0; i < io_num.n_output; i++) {
    uint32_t MaxClass[1];
    float    fMaxProb[1];
    float*   buffer = (float*)outputs[i].buf;
    uint32_t sz     = outputs[i].size / 4;

    int len = sizeof(buffer) / sizeof(buffer[0]);
    // printf("%3d: %3d: \n", outputs[i].size,len);

    rknn_GetTop(buffer, fMaxProb, MaxClass, sz, 1);

    printf(" --- Top1 ---\n");
    printf("%3d: %8.6f\n", MaxClass[0], fMaxProb[0]);   //返回
    
  }

  // Release rknn_outputs
  rknn_outputs_release(ctx, 1, outputs);
  return 0;
}

int release(){
  // Release
  if (ctx > 0)
  {
    rknn_destroy(ctx);
  }
  if (model) {
    free(model);
  }
  return 0;
}
