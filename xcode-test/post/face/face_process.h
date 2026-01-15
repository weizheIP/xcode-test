#ifndef FACE_PROCESS
#define FACE_PROCESS

#include <iostream>
#include <mutex>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/types_c.h"
#include "opencv2/videoio.hpp"
#include <faiss/IndexFlat.h>

#include "base64.h"
#include "json.h"

#include "post/queue.h"
// #include "prefer/get_videoffmpeg.hpp"
#include <thread>


#include "Track/inc/SORT.h"
#include "face_db.h"

#include <map>

using idx_t = faiss::idx_t;

struct sendMsg {
    std::string msg;
    timeval tvN[6];
};

struct InferOutputMsg {
    uint32_t frameWidth;
    uint32_t frameHeight;

    cv::Mat rawImage;

};

struct processInfo
{
    int isAlarm;
    std::string recName;
    bool isRecing;
};

class Postprocess_comm{
public:

    Postprocess_comm(std::string xx);

    ~Postprocess_comm();

    int oldNum = 0;
    int n = 0;
    time_t time_interval;
    int is_alarmed = 0;
    int oldMode = 0;
    float stranger_alarm_time = -1;
    float oldtime = -1;//旧的时间间隔
    time_t time_start_dict1 = 0;
    time_t ntime = 0;
    FILE *josnF;
    std::string rtsp_path;
    char recNameBuf[200]={0};
    bool status = 0;
    std::shared_ptr<BlockingQueue<sendMsg>> send_list;

    std::map<idx_t,time_t> faceIdTimeMap;

    std::string makeSendMsg(std::shared_ptr<InferOutputMsg> data,Json::Value jresult,Json::Value jwarn,Json::Value roi,Json::Value jstranger,int alarm);
    std::string Process(std::shared_ptr<InferOutputMsg> data,int &isAlarm);


    
    std::thread *t_sender = nullptr;
    void sendThread();

private:
    faceRec * facetool;
    
};


#endif