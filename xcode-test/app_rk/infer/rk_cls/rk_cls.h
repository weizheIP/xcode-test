#ifndef RK_CLS
#define RK_CLS

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "rknn_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include <fstream>
#include <iostream>

#include <malloc.h>
#include <iostream>


#include "common/license/license.h"

struct clsInfo
{
    int lableIndex;
    float score;
};

class rknn_cls
{
public:
    rknn_cls();
    ~rknn_cls();
    int init(const char* model_path);
    int detect(cv::Mat &orig_img,std::vector<clsInfo> &info);
    int release();
private:
    rknn_context ctx = 0;
    rknn_input_output_num io_num;
    const int MODEL_IN_WIDTH    = 224;
    const int MODEL_IN_HEIGHT   = 224;
};

#endif