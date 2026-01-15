#ifndef FACE_EXP_H
#define FACE_EXP_H
#include "common/license/license.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;
using namespace Ort;

class FaceExpression
{
public:
    FaceExpression(string model_path){
        unsigned char *outdata;
        FILE *f = fopen(model_path.c_str(),"r");
        fseek(f,0,SEEK_END);
        size_t size = ftell(f);
        fseek(f,0,SEEK_SET);
        unsigned char* buff = (unsigned char*)malloc(size);
        fread(buff,size,1,f);
        fclose(f);        
        get_model_decrypt_value(buff,outdata,size);
    
        // 初始化 ONNX Runtime C++ API 环境
        Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "example"};
        Ort::SessionOptions session_options{};
        // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
        // session = new Session(env, "/media/ps/data1/liuym/github/face/rec_qua_age/facial_expression/ResEmoteNet/ResEmoteNet_Checkpoints/rafdb_model.onnx", session_options);  // 加载模型
        session = new Session(env, model_path.c_str(), session_options);  // 加载模型
        // session = new Session(env, buff,size, session_options);  // 加载模型
    
        free(buff);
    }
	int feature(Mat frame){
         // 读取和预处理图像
        cv::Mat image;
        // cv::resize(frame, image, cv::Size(64, 64));  // 缩放到指定大小
        // cv::cvtColor(image, image, cv::COLOR_BGR2GRAY); // 将图像颜色通道由BGR转换为RGB
        // std::vector<cv::Mat> channels_vec{image, image, image};
        // cv::merge(channels_vec, image);

        cv::resize(frame, image, cv::Size(224, 224));  // 2 DAN
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        image.convertTo(image, CV_32FC3, 1.0 / 255.0); // 转换图像数据类型为32位浮点型 // 归一化到[0, 1]范围
        cv::subtract(image, cv::Scalar(0.485, 0.456, 0.406), image); // 减去均值
        cv::divide(image, cv::Scalar(0.229, 0.224, 0.225), image); // 除以方差
        cv::Mat inputBolb = cv::dnn::blobFromImage(image);


        // 将归一化后的图片数据存储到 OrtValue 中
        // vector<int64_t> input_shape = {1, 3, 64, 64};
        vector<int64_t> input_shape = {1, 3, 224, 224};
        std::vector<float> input_data(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 0.f);
        memcpy(input_data.data(), inputBolb.ptr<float>(0), input_data.size() * sizeof(float));
        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_data.data(), input_data.size(), input_shape.data(),
                                                                input_shape.size());

        // 执行推理操作
        const char* input_name = "input";
        const char* output_name = "output";
        vector<Ort::Value> output_tensors = session->Run(Ort::RunOptions{}, &input_name, &input_tensor, 1, &output_name, 1);

        // 处理结果
        vector<float> output_data(output_tensors.front().GetTensorTypeAndShapeInfo().GetShape().at(1));
        memcpy(output_data.data(), output_tensors.front().GetTensorMutableData<float>(), output_data.size() * sizeof(float));

        auto max_it = std::max_element(output_data.begin(), output_data.end());
        int max_index = std::distance(output_data.begin(), max_it);
        
        return max_index;
    }
private:
    Ort::Session *session = nullptr;
};



#endif