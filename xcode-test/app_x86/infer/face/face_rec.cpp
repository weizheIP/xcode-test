#include "face_rec.h"
#include "common/license/license.h"
#include <stdexcept>

FaceRecognizer::FaceRecognizer(string model_path) {
    unsigned char *outdata = nullptr;
    FILE *f = fopen(model_path.c_str(),"r");
    if (!f) {
        throw std::runtime_error("Failed to open model file: " + model_path);
    }
    fseek(f,0,SEEK_END);
    size_t size = ftell(f);
    fseek(f,0,SEEK_SET);
    unsigned char* buff = (unsigned char*)malloc(size);
    if (!buff) {
        fclose(f);
        throw std::runtime_error("Failed to allocate memory for model");
    }
    fread(buff,size,1,f);
    fclose(f);        
    
    // Decrypt the model
    if (get_model_decrypt_value(buff, outdata, size) != 0 || !outdata) {
        free(buff);
        throw std::runtime_error("Failed to decrypt model");
    }

    // 初始化 ONNX Runtime C++ API 环境
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "example"};
    Ort::SessionOptions session_options{};
    
    // Try to add CUDA provider, but fall back to CPU if it fails
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    if (status) {
        // CUDA provider failed, use CPU instead
        std::cerr << "[WARNING] CUDA provider failed, falling back to CPU" << std::endl;
        OrtReleaseStatus(status);
    }
    
    // Create session with decrypted model data
    try {
        session = new Session(env, outdata, size, session_options);
    } catch (const Ort::Exception& e) {
        free(buff);
        throw std::runtime_error("Failed to create ONNX session: " + std::string(e.what()));
    }

    // Free the original buffer, but keep outdata for the session
    free(buff);
}

FaceRecognizer::~FaceRecognizer() {
    if (session) {
        delete session;
        session = nullptr;
    }
}

vector<float> FaceRecognizer::feature(cv::Mat src) {
    // 读取和预处理图像
    cv::Mat image;
    cv::resize(src, image, cv::Size(112, 112));  // 缩放到指定大小
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // 将图像颜色通道由BGR转换为RGB
    image.convertTo(image, CV_32FC3, 1.0 / 255.0); // 转换图像数据类型为32位浮点型 // 归一化到[0, 1]范围
    cv::subtract(image, cv::Scalar(0.5, 0.5, 0.5), image); // 减去均值
    cv::divide(image, cv::Scalar(0.5, 0.5, 0.5), image); // 除以方差
    cv::Mat inputBolb = cv::dnn::blobFromImage(image);


    // 将归一化后的图片数据存储到 OrtValue 中
    vector<int64_t> input_shape = {1, 3, 112, 112};
    std::vector<float> input_data(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3], 0.f);
    memcpy(input_data.data(), inputBolb.ptr<float>(0), input_data.size() * sizeof(float));
    auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(allocator_info, input_data.data(), input_data.size(), input_shape.data(),
                                                              input_shape.size());

    // 执行推理操作
    const char* input_name = "input.1";
    const char* output_name = "516";
    vector<Ort::Value> output_tensors = session->Run(Ort::RunOptions{}, &input_name, &input_tensor, 1, &output_name, 1);

    // 处理结果
    vector<float> output_data(output_tensors.front().GetTensorTypeAndShapeInfo().GetShape().at(1));
    memcpy(output_data.data(), output_tensors.front().GetTensorMutableData<float>(), output_data.size() * sizeof(float));
    
    return output_data;
}