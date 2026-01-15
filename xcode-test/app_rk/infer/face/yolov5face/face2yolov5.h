#ifndef FACE2YOlOV5
#define FACE2YOlOV5

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define _BASETSD_H

#include "RgaUtils.h"
#include "im2d.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "postprocess.h"
#include "rga.h"
#include "rknn_api.h"
#include "post/base.h"

// typedef struct{
//     float x;
//     float y;
//     float w;
//     float h;
//     float conf;
//     std::vector<cv::Point> keypoint;
// }Yolov5Face_BoxStruct;

class Face_Detector2
{
private:
    char *model_name;
    rknn_context ctx;

    rknn_input_output_num io_num;
	int g_model_w=1280,g_model_h=1280;

public:
    Face_Detector2(char* model_name_):model_name(model_name_){
        ;
    }
    ~Face_Detector2();
    int init();
    int detect_face(cv::Mat orig_img,std::vector<Yolov5Face_BoxStruct> &face_result,float nms_thresh,float box_thresh);

    
};

#endif