#ifndef MODEL_BASE
#define MODEL_BASE

#if defined RK
#include "inferModel/rkModel.h"
// #include "prefer/get_videoffmpeg.hpp"
#endif



#if defined ATLAS
#include "inferModel/atlasModel.h"
#endif

#if defined SOPHON
#include "inferModel/sophonModel.h"
#endif


#if defined JETSON
#include "inferModel/jetsonModel.h"
// #include "prefer/get_video.hpp"
#endif


#if defined X86_TENSORRT
#include "inferModel/trtModel.h"
#endif

#if defined X86_TS
#include "inferModel/tsModel.h"
#endif

#if defined(X86_TENSORRT) && defined(SEG)
#endif



// 模型和解码定义
// //
// // 抽象类
// class Platform_Yolo {
// public:
//     virtual void init(std::string model_path) = 0; // 纯虚函数
//     virtual std::vector<Object> infer(cv::Mat img) = 0; // 纯虚函数
//     virtual std::vector<std::vector<BoxInfo>> infer_batch(std::vector<cv::Mat> imgs) = 0;

// };
// class Atlas : public Platform_Yolo {
// public:
//     void init() override {
//         // Atlas平台的初始化代码
//     }

//     void infer() override {
//         // Atlas平台的推理代码
//     }
// };



#endif