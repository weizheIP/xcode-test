#ifndef FACE_REC_H
#define FACE_REC_H


#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;
using namespace Ort;

class FaceRecognizer
{
public:
	FaceRecognizer(string model_path);
	~FaceRecognizer();
	vector<float> feature(Mat frame);
private:
    Ort::Session *session = nullptr;
};



#endif