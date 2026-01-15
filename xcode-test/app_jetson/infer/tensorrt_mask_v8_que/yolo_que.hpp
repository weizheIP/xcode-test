#ifndef ___YOLO_2__
#define ___YOLO_2__

#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "NvInferVersion.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "muti2.cuh"
#include "post/queue.h"
#include <thread>
#include <unistd.h>
#include <time.h>
#include <math.h>

// #include "infer/yolo_commom.h"
#include "post/base.h"


// #define __TEST__
// #ifdef __TEST__
//     #define DRAW_MASK
//     // #define _RAW_
// #else
//     // #define DRAW_MASK
//     #define _RAW_
// #endif


using namespace nvinfer1;



typedef struct{
    std::vector<Object> objects;
    cv::Mat img;
    cv::Mat maskProposals;
     cv::Mat protos;
    // std::vector<float> pad_info;
    // float* prob = nullptr;
    // float* prob_mask = nullptr;
}Det_Struct;

// void DrawPred(cv::Mat& img, std::vector<Object> result);
class YOLO_QUE
{
    public:
        int runCnt_;
        bool bRun_Decode = true;
        // BlockingQueue<Det_Struct> queue_det_list;
        BlockingQueue<Det_Struct> queue_det_ForDet_list;
        BlockingQueue<Det_Struct> queue_result_list;

        std::thread* th_Decode = nullptr;

        Det_Struct detect_img(cv::Mat& img,bool& bGet);
    public:
        YOLO_QUE(std::string engine_file_path);
        virtual ~YOLO_QUE();
        // void detect_img(std::string image_path);
        // void detect_video(std::string video_path);
        // cv::Mat static_resize(cv::Mat& img);
        // float* blobFromImage(cv::Mat& img);
        void doInference(IExecutionContext& context, cv::Mat img, float* output,float* output_mask, const int output_size, cv::Size input_shape ,float &scale);
        std::vector<Object> detect_img(cv::Mat img);
        void InitInfer(IExecutionContext& context);
    private:
        int INPUT_W = 1280;
        int INPUT_H = 1280;
        int dx = 0;
        int wh = 0;
        const char* INPUT_BLOB_NAME = "images";
        // const char* OUTPUT_BLOB_NAME = "output";
        const char* OUTPUT_BLOB_NAME = "output0";
        const char* OUTPUT_BLOB_NAME_MASK = "output1";
        // float** prob,**prob_mask;
        // BlockingQueue<float*> prob_list,prob_mask_list;
        int output_size = 1;
        int output_size_mask = 1;
        ICudaEngine* engine;
        IRuntime* runtime;
        IExecutionContext* context;
        int inputIndex=0,outputIndex_det=1,outputIndex_seg=2;
    private:
        // ICudaEngine& engine;
        cudaStream_t stream;
        void* buffers[3] = {nullptr, nullptr, nullptr}; // input和output
        void* imgbuffer[3] = {nullptr, nullptr, nullptr}; // 输入的三通道
};

#endif
