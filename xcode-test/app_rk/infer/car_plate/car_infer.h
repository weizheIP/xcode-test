#ifndef __CAR_INFER
#define __CAR_INFER


#include "rknn_api.h"

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/videoio.hpp"

#include <vector>
#include <string>
#include <map>
#include <stdint.h>
#include <iostream>
#include <sys/time.h>


#include "postprocess.h"

#include "post/base.h"




// struct CarBoxInfo
// {
//     int x1;
//     int y1;
//     int x2;
//     int y2;
//     int label;
//     float score;
//     std::vector<float> landmark;
// };

class CarPlateDet
{
public:
    CarPlateDet(char* modelPath,int type = 0);
    ~CarPlateDet();
    int init(int coreIndex);
    int detect(cv::Mat &img,std::vector<CarBoxInfo>& boxes);
private:
    

float _NMS_THRESH;
float _BOX_THRESH;


int _modelInWidth;
int _modelInHeight;
int _modelInChannels;

rknn_context _ctx = 0;
rknn_input_output_num _ioNum;

unsigned char* _model;
int _modelLen = 0;
char* _modelPath;

};




class CarPlateRec
{
public:
    CarPlateRec(char* name);
    ~CarPlateRec();
    int init(int coreIndex);
    std::string get_car_plate_code(cv::Mat& img);

private:

int _modelInWidth;
int _modelInHeight;
int _modelInChannels;

rknn_context _ctx = 0;
rknn_input_output_num _ioNum;

unsigned char* _model;
int _modelLen = 0;
char* _modelPath;

std::vector<std::string> _keys;
};

class CarPlateColor
{
public:
CarPlateColor(char* name);
~CarPlateColor();
int init();
int get_car_plate_color(cv::Mat& img);

private:

rknn_context _ctx = 0;
rknn_input_output_num _ioNum;

int _modelInWidth;
int _modelInHeight;
int _modelInChannels;

unsigned char* _model;
int _modelLen = 0;
char* _modelPath;


};


#endif