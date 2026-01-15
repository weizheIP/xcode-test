#ifndef CV_TR_H
#define CV_TR_H

#include "NvInfer.h"
#include "cuda_runtime_api.h"
// #include "NvInferPlugin.h"
#include "logging.h"
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <string>
#include <vector>

using namespace nvinfer1;

#include "post/base.h"

// struct Object
// {
//     // cv::Rect_<float> rect;
//     cv::Rect rect;
//     int label;
//     float prob;
// };

// typedef struct BoxInfo
// {
// 	float x1;
// 	float y1;
// 	float x2;
// 	float y2;
// 	float score;
// 	int label;
// } BoxInfo;


class TRtDet
{
public:
    TRtDet(std::string path, int classNum, int ModelSize, std::string appName):
    _modelPath(path),
    _modelW(ModelSize),
    _modelH(ModelSize),
    _classNum(classNum),
    _appName(appName),
    _engine(nullptr),
    _context(nullptr),
    _prob(nullptr)
    {}

    ~TRtDet();
        
    int init(int gpu_device=0);
    int det(cv::Mat &img, std::vector<BoxInfo> &boxes);
    
private:


    void doInference(IExecutionContext* context, float* input, float* output, int batchSize);

    

    
    // std::vector<int> _useTimeArr;
    // std::vector<int> _useTimeArr2;
    // std::vector<int> _useTimeArr3;
    nvinfer1::IExecutionContext *_context;
    nvinfer1::ICudaEngine* _engine;
    nvinfer1::IRuntime* _runtime;
    std::string _modelPath;
    int _modelH;
    int _modelW;
    int _classNum;
    float *_prob;
    float *_data;
    int _outPutSize;
    std::string _appName;
    int Num_box = 34000; 
    
};


#endif