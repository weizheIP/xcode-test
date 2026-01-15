#ifndef ___YOLO__
#define ___YOLO__

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
#include "post/queue.h"
#include <thread>
#include <unistd.h>
#include <time.h>

#define __TEST__
#ifdef __TEST__
    //#define DRAW_MASK
    // #define _RAW_
#else
    // #define DRAW_MASK
    #define _RAW_
#endif


using namespace nvinfer1;
static Logger gLogger;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;	 // Non-maximum suppression threshold
	float objThreshold;	 // Object Confidence threshold
};

#include "post/base.h"

// typedef struct BoxInfo
// {
// 	float x1;
// 	float y1;
// 	float x2;
// 	float y2;
// 	float score;
// 	int label;
// } BoxInfo;

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
// };

typedef struct{
    std::vector<Object> objects;
    cv::Mat img;
    std::vector<float> pad_info;
    float* prob = nullptr;
    float* prob_mask = nullptr;
}Det_Struct;

void DrawPred(cv::Mat& img, std::vector<Object> result);
class YOLO
{
    public:

        bool bRun_Decode = true;
        BlockingQueue<Det_Struct> queue_det_list;
        BlockingQueue<Det_Struct> queue_det_ForDet_list;
        BlockingQueue<Det_Struct> queue_result_list;
        std::thread* th_Decode = nullptr;
        Det_Struct detect_img(cv::Mat& img,bool& bGet);
    public:
        YOLO(std::string engine_file_path);
        virtual ~YOLO();
        void detect_img(std::string image_path);
        void detect_video(std::string video_path);
        cv::Mat static_resize(cv::Mat& img);
        float* blobFromImage(cv::Mat& img);
        void doInference(IExecutionContext& context, cv::Mat img, float* output,float* output_mask, const int output_size, cv::Size input_shape ,float &scale);
        std::vector<Object> detect_img(cv::Mat& img);
        void InitInfer(IExecutionContext& context);
    private:     
        int INPUT_W = 1280;
        int INPUT_H = 1280;
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
        int inputIndex,outputIndex_det,outputIndex_seg;
    private:
        // ICudaEngine& engine;
        cudaStream_t stream;
        void* buffers[3]; // input和output
        void* imgbuffer[3]; // 输入的三通道
};

#endif
