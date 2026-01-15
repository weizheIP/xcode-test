#ifndef FACE_DET_H
#define FACE_DET_H

#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// #include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#include "post/base.h"


using namespace std;
using namespace cv;
using namespace Ort;

struct Net_config
{
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	string modelpath;
};

// typedef struct PointInfo
// {
// 	Point pt;
// 	float score;
// } PointInfo;


// typedef struct BoxInfo
// {
// 	float x1;
// 	float y1;
// 	float x2;
// 	float y2;
// 	float score;
// 	PointInfo kpt1;
// 	PointInfo kpt2;
// 	PointInfo kpt3;
// 	PointInfo kpt4;
// 	PointInfo kpt5;
// } BoxInfo;

class YOLOV7_face
{
public:
	YOLOV7_face(Net_config config);
	vector<Yolov5Face_BoxStruct> detect(Mat& frame);
	vector<Yolov5Face_BoxStruct> detect_v8(Mat& frame);
private:
	int inpWidth;
	int inpHeight;
	int nout;
	int num_proposal;

	float confThreshold;
	float nmsThreshold;
	vector<float> input_image_;
	void normalize_(Mat img);
	void nms(vector<Yolov5Face_BoxStruct>& input_boxes);
	bool has_postprocess;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "YOLOV7_face");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};



#endif