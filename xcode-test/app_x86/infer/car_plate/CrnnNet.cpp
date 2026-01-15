#include "CrnnNet.h"
// #include "OcrUtils.h"
#include <fstream>
#include <numeric>
#include "common/license/license.h"

extern std::string sTypeName;

CrnnNet::CrnnNet() {}

CrnnNet::~CrnnNet() {
    delete session;
    free(inputName);
    free(outputName);
}

void CrnnNet::setNumThread(int numOfThread) {
    numThread = numOfThread;
    //===session options===
    // Sets the number of threads used to parallelize the execution within nodes
    // A value of 0 means ORT will pick a default
    //sessionOptions.SetIntraOpNumThreads(numThread);
    //set OMP_NUM_THREADS=16

    // Sets the number of threads used to parallelize the execution of the graph (across nodes)
    // If sequential execution is enabled this value is ignored
    // A value of 0 means ORT will pick a default
    sessionOptions.SetInterOpNumThreads(numThread);

    // Sets graph optimization level
    // ORT_DISABLE_ALL -> To disable all optimizations
    // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
    // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
    // ORT_ENABLE_ALL -> To Enable All possible opitmizations
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
}


void getInputName(Ort::Session *session, char *&inputName) {
    size_t numInputNodes = session->GetInputCount();
    if (numInputNodes > 0) {
        Ort::AllocatorWithDefaultOptions allocator;
        {
            char *t = session->GetInputName(0, allocator);
            inputName = strdup(t);
            allocator.Free(t);
        }
    }
}

void getOutputName(Ort::Session *session, char *&outputName) {
    size_t numOutputNodes = session->GetInputCount();
    if (numOutputNodes > 0) {
        Ort::AllocatorWithDefaultOptions allocator;
        {
            char *t = session->GetOutputName(0, allocator);
            outputName = strdup(t);
            allocator.Free(t);
        }
    }
}

std::vector<float> substractMeanNormalize(cv::Mat &src, const float *meanVals, const float *normVals) {
    auto inputTensorSize = src.cols * src.rows * src.channels();
    std::vector<float> inputTensorValues(inputTensorSize);
    size_t numChannels = src.channels();
    size_t imageSize = src.cols * src.rows;

    for (size_t pid = 0; pid < imageSize; pid++) {
        for (size_t ch = 0; ch < numChannels; ++ch) {
            float data = (float) (src.data[pid * numChannels + ch] * normVals[ch] - meanVals[ch] * normVals[ch]);
            inputTensorValues[ch * imageSize + pid] = data;
        }
    }
    return inputTensorValues;
}

void CrnnNet::initModel(const std::string &pathStr, const std::string &keysPath) {

    unsigned char *outdata;
    FILE *f = fopen(pathStr.c_str(),"r");
    fseek(f,0,SEEK_END);
    size_t size = ftell(f);
    fseek(f,0,SEEK_SET);
    unsigned char* buff = (unsigned char*)malloc(size);
    fread(buff,size,1,f);
    fclose(f);        
    get_model_decrypt_value(buff,outdata,size);


    // session = new Ort::Session(env, pathStr.c_str(), sessionOptions);
    session = new Ort::Session(env, buff,size, sessionOptions);  // 加载模型
    free(buff);

    getInputName(session, inputName);
    getOutputName(session, outputName);

    // //load keys
    // std::ifstream in(keysPath.c_str());
    // std::string line;
    // if (in) {
    //     while (getline(in, line)) {// line中不包括每行的换行符
    //         keys.push_back(line);
    //     }
    // } else {
    //     printf("The keys.txt file was not found\n");
    //     return;
    // }
    // /*if (keys.size() != 6623) {
    //     fprintf(stderr, "missing keys\n");
    // }*/
    // keys.insert(keys.begin(), "#");
    // keys.emplace_back(" ");



    // 023456789ABWTYKHMX // 钢印 18个
    if( sTypeName == "cv_ocr_detection"){
    keys={  "#", 
            "0", "2", "3", "4", "5", "6", "7", "8", "9",  
            "A", "B", "W", "T", "Y", "K", "H", "M", "X"
        };
    }else{
    // 车牌74个
    keys={  "#", 
        "A", "B", "C", "D", "E", "F", "G", 
        "H", "J", "K", "L", "M", "N", 
        "P", "Q", "R", "S", "T", 
        "U", "V", "W", "X", "Y", "Z", 
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
        "皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","西","陕","甘","青","宁","新","港","澳","学","警"
    };
    }

    // printf("total keys size(%lu)\n", keys.size());
}

template<class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

TextLine CrnnNet::scoreToTextLine(const std::vector<float> &outputData, int h, int w) {
    int keySize = keys.size();
    std::string strRes;
    std::vector<float> scores;
    int lastIndex = 0;
    int maxIndex;
    float maxValue;

    for (int i = 0; i < h; i++) {
        maxIndex = int(argmax(&outputData[i * w], &outputData[(i + 1) * w - 1]));
        maxValue = float(*std::max_element(&outputData[i * w], &outputData[(i + 1) * w - 1]));

        if (maxIndex > 0 && maxIndex < keySize && (!(i > 0 && maxIndex == lastIndex))) {
            scores.emplace_back(maxValue);
            strRes.append(keys[maxIndex]);
        }
        lastIndex = maxIndex;
    }
    return {strRes, scores};
}

TextLine CrnnNet::getTextLine(const cv::Mat &src) {

    // imgH = 32;
    // imgW = 320; //训练时
        
    // imgW = int(32 * wh_ratio);

    // float ratio = float(img.cols) / float(img.rows);
    // int resize_w, resize_h;

    // if (ceilf(imgH * ratio) > imgW)
    //     resize_w = imgW;
    // else
    //     resize_w = int(ceilf(imgH * ratio));
        
    // cv::resize(img, resize_img, cv::Size(resize_w, imgH), 0.f, 0.f,
    //             cv::INTER_LINEAR);
    // cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
    //                     int(imgW - resize_img.cols), cv::BORDER_CONSTANT,
    //                     {127, 127, 127});


    float scale = (float) dstHeight / (float) src.rows;
    int dstWidth = int((float) src.cols * scale); //这里宽度要改，看python版本

    cv::Mat srcResize;
    resize(src, srcResize, cv::Size(dstWidth, dstHeight));

    std::vector<float> inputTensorValues = substractMeanNormalize(srcResize, meanValues, normValues);

    std::array<int64_t, 4> inputShape{1, srcResize.channels(), srcResize.rows, srcResize.cols};

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(),
                                                             inputTensorValues.size(), inputShape.data(),
                                                             inputShape.size());
    assert(inputTensor.IsTensor());

    auto outputTensor = session->Run(Ort::RunOptions{nullptr}, &inputName, &inputTensor, 1, &outputName, 1);

    assert(outputTensor.size() == 1 && outputTensor.front().IsTensor());

    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1,
                                          std::multiplies<int64_t>());

    float *floatArray = outputTensor.front().GetTensorMutableData<float>();
    std::vector<float> outputData(floatArray, floatArray + outputCount);
    return scoreToTextLine(outputData, outputShape[1], outputShape[2]);
}

// std::vector<TextLine> CrnnNet::getTextLines(std::vector<cv::Mat> &partImg, const char *path, const char *imgName) {
//     int size = partImg.size();
//     std::vector<TextLine> textLines(size);
// #ifdef __OPENMP__
// #pragma omp parallel for num_threads(numThread)
// #endif
//     for (int i = 0; i < size; ++i) {
//         //OutPut DebugImg
//         if (isOutputDebugImg) {
//             std::string debugImgFile = getDebugImgFilePath(path, imgName, i, "-debug-");
//             saveImg(partImg[i], debugImgFile.c_str());
//         }

//         //getTextLine
//         double startCrnnTime = getCurrentTime();
//         TextLine textLine = getTextLine(partImg[i]);
//         double endCrnnTime = getCurrentTime();
//         textLine.time = endCrnnTime - startCrnnTime;
//         textLines[i] = textLine;
//     }
//     return textLines;
// }