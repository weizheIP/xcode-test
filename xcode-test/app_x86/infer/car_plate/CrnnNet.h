#ifndef __OCR_CRNNNET_H__
#define __OCR_CRNNNET_H__

// #include "OcrStruct.h"
#include "onnxruntime_cxx_api.h"
#include <opencv2/opencv.hpp>


struct TextLine {
    std::string text;
    std::vector<float> charScores;
    double time;
};

class CrnnNet {
public:

    CrnnNet();

    ~CrnnNet();

    void setNumThread(int numOfThread);

    void initModel(const std::string &pathStr, const std::string &keysPath);

    // std::vector<TextLine> getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName);
    TextLine getTextLine(const cv::Mat &src);

private:
    bool isOutputDebugImg = false;
    Ort::Session *session;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "CrnnNet");
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();  //gpu
    OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);


    int numThread = 0;

    char *inputName;
    char *outputName;

    const float meanValues[3] = {127.5, 127.5, 127.5};
    const float normValues[3] = {1.0 / 127.5, 1.0 / 127.5, 1.0 / 127.5};
    const int dstHeight = 32;
    // const int dstHeight = 48;

    std::vector<std::string> keys;

    TextLine scoreToTextLine(const std::vector<float> &outputData, int h, int w);

    
};


#endif //__OCR_CRNNNET_H__
