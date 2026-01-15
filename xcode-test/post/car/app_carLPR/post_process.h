


#ifndef _CAR_POST_PROCESS
#define _CAR_POST_PROCESS


#include <iostream>
#include <mutex>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include "base64.h"
#include "json.h"
#include "post/queue.h"
#include "post/base.h"
#include <thread>
#include "Track/inc/SORT.h"
#include <map>
using namespace std;

struct message_img
{
    string videoIndex;
    cv::Mat frame;
};

struct message_det
{
    string videoIndex;
    cv::Mat frame;
    vector<BoxInfo> detectData;
};

struct message_lpr_img
{
    string videoIndex;
    cv::Mat frame;
    BoxInfo inBox;
};

struct message_lpr
{
    string videoIndex;
    cv::Mat frame;
    BoxInfo detectData;
    string lprCode;
};

static int chooseBestCarPlateIndex(std::vector<CarBoxInfo>& vec,int boxHeight)
{
    int ans = -1;
    float maxScore = 0;
    for(int i = 0; i < vec.size(); i++)
    {
        if(vec[i].y1 < (boxHeight/3)) //车牌的位置还没变换，车牌的位置必须大于车框的高度的三分之一
            continue;
        if(vec[i].score > maxScore)
        {
            maxScore = vec[i].score;
            ans = i;
        }
    }

    return ans;
}

template<typename T>
T t_get_vec_aver(std::vector<T> vec) {
    if(vec.size() == 0)
        return (T)0;
    T sum = 0;
    for(auto item:vec)
    {
        sum+=item;
    }
    return sum/vec.size();
}

struct sendMsg {
    std::string msg;
    timeval tvN[6];
    time_t msgId;
    bool isA;
};

struct InferOutputMsg {
    std::vector<BoxInfo> result;
    cv::Mat rawImage;
};

struct processInfo
{
    int isAlarm;
    std::string recName;
    bool isRecing;
    std::vector<std::string> fileNameVec;
};



class Postprocess_comm{
public:

    Postprocess_comm(std::string rtsp_path,std::string typeName);
    ~Postprocess_comm();

    std::string c_typeName;
    std::string rtsp_path;
   
    // 时间间隔
    std::map<std::string,std::time_t> time_start_dict; //持续时间
    int n = 0;    
    float oldtime = -1;//旧的时间间隔
    
    // 录像文件
    char recNameBuf[200]={0};

    // 违停
    bool illegal_parking_detection(std::string ID,SORT::TrackingBox frameTrackingResult);
    std::map<string, std::vector<SORT::TrackingBox>> ID_FireSmokeListHis;
    std::map<string, time_t> ID_T;
    std::vector<SORT::TrackingBox> oldTrackBoxs;

    std::string makeSendMsg(std::shared_ptr<InferOutputMsg> data,Json::Value jresult,Json::Value roi,Json::Value config,Json::Value jwarn,int alarm, bool isRecing);
    std::string process(std::shared_ptr<InferOutputMsg> data, processInfo &info);   

    // 跟踪
    int fi=0;
    SORT* trackPtr_; 
    
    // 发送
    bool status = 0; //发送线程的状态
    std::shared_ptr<BlockingQueue<sendMsg>> send_list;
    std::thread *t_sender = nullptr;
    void sendThread();   
    
};


#endif