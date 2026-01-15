#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "muti.cuh"

// #include "infer/yolo_commom.h"
#include "post/base.h"

using namespace nvinfer1;


class YOLO
{
    public:
        YOLO(std::string engine_file_path,int w = 1280 ,int h = 1280);
        virtual ~YOLO();
        void detect_img(std::string image_path);
        void detect_video(std::string video_path);
        cv::Mat static_resize(cv::Mat& img);
        float* blobFromImage(cv::Mat& img);
        void doInference(IExecutionContext& context, cv::Mat img, float* output, const int output_size, cv::Size input_shape ,float &scale);
        std::vector<BoxInfo> detect_img(cv::Mat img);
        std::vector<Object> detect_mask(cv::Mat& img);
    private:
        int INPUT_W = 1280;
        int INPUT_H = 1280;
        const char* INPUT_BLOB_NAME = "images";
        // const char* OUTPUT_BLOB_NAME = "output";
        const char* OUTPUT_BLOB_NAME = "output0";
        float* prob;
        int output_size = 1;
        int inputIndex=0;
        int outputIndex =1;
        ICudaEngine* engine;
        int modleClassNum = 0;
        IRuntime* runtime;
        IExecutionContext* context;
};
