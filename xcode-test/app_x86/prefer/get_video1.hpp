#pragma once
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/videoio.hpp"
#include <vector>
#include <thread>
#include <iostream>
#include <stdio.h>
#include <mutex>
#include <post/queue.h>
#include "unistd.h"
#include <iostream>
#include "string.h"
#include "sys/time.h"
#include "cuda_runtime_api.h"

// static std::mutex mtx_cv;
static std::string check_codec_type(std::string rtspUrl)
{
    // std::lock_guard<std::mutex> lock(mtx_cv);
    cv::VideoCapture cap(rtspUrl, cv::CAP_FFMPEG);
    if (!cap.isOpened())
    {
        std::cout << "[ERROR] Failed to open RTSP stream."<<rtspUrl << std::endl;
        cap.release();
        return "";
    }
    // 获取视频编码类型
    int codec = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
    std::string codecString = std::string(reinterpret_cast<char *>(&codec), 4);
    cap.release();
    return codecString;
}
static void sw_resize(cv::Mat &tempImg1)
{
    #ifdef Model640
    int size = 640;
    #else
    int size = 1280;
    #endif
    if(tempImg1.cols != size && !tempImg1.empty())
    {   cv::Mat dstimg;
        int w = size, h = 0;
        h = tempImg1.rows * size /tempImg1.cols;
        //std::cout<<tempImg1.cols<<tempImg1.rows<<"-"<<dstw<<":"<<dsth<<std::endl;
        if(h%2)
            h++;
        cv::resize(tempImg1, dstimg, cv::Size(w, h), cv::INTER_AREA);
        tempImg1 = dstimg.clone();
        dstimg.release();
    }
    
}
static std::string get_rtsp_name(std::string videoPath, int latency, int width, int height, int codeType)
{
    std::string ans;
    if (videoPath.compare(0, 4, "rtsp") == 0)
    {
        // tcp
        #ifdef DECNOTRESIZE
        if (codeType)
            ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp tcp-timeout=5 timeout=5 ! rtph264depay ! h264parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
        else
            ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp tcp-timeout=5 timeout=5 ! rtph265depay ! h265parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
        #else
        if (codeType)
            ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp tcp-timeout=5 timeout=5 ! rtph264depay ! h264parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
        else
            ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp tcp-timeout=5 timeout=5 ! rtph265depay ! h265parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
        #endif

        // #ifdef DECNOTRESIZE
        // if (codeType)
        //     ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp tcp-timeout=5 timeout=5 ! rtph264depay ! queue max-size-buffers=3 leaky=1 ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, format=(string)BGRx ! appsink max-buffers=1 drop=true sync=false";
        // else
        //     ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp tcp-timeout=5 timeout=5 ! rtph265depay ! queue max-size-buffers=3 leaky=1 ! h265parse ! avdec_h265 ! videoconvert! video/x-raw, format=(string)BGRx ! appsink max-buffers=1 drop=true sync=false";
        // #else
        // if (codeType)
        //     ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp tcp-timeout=5 timeout=5 ! rtph264depay ! queue max-size-buffers=3 leaky=1 ! h264parse ! avdec_h264 ! videoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! appsink max-buffers=1 drop=true sync=false";
        // else
        //     ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp tcp-timeout=5 timeout=5 ! rtph265depay ! queue max-size-buffers=3 leaky=1 ! h265parse ! avdec_h265 ! videoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! appsink max-buffers=1 drop=true sync=false";
        // #endif

        // 错误
        // #ifdef DECNOTRESIZE
        // if (codeType)
        //     ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp tcp-timeout=5 timeout=5 ! rtph264depay ! queue max-size-buffers=3 leaky=1 ! h264parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! videorate ! video/x-raw, framerate=1/1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
        // else
        //     ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp tcp-timeout=5 timeout=5 ! rtph265depay ! queue max-size-buffers=3 leaky=1 ! h265parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! videorate ! video/x-raw, framerate=1/1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
        // #else
        // if (codeType)
        //     ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp tcp-timeout=5 timeout=5 ! rtph264depay ! queue max-size-buffers=3 leaky=1 ! h264parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! videorate ! video/x-raw, framerate=1/1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
        // else
        //     ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp tcp-timeout=5 timeout=5 ! rtph265depay ! queue max-size-buffers=3 leaky=1 ! h265parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! videorate ! video/x-raw, framerate=1/1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
        // #endif


        // #ifdef DECNOTRESIZE
        // if (codeType)
        //     ans = std::string("rtspsrc location=") + videoPath + " latency=0 buffer-mode=none protocols=tcp tcp-timeout=5 timeout=5 ! rtph264depay ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=1 ! h264parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
        // else
        //     ans = std::string("rtspsrc location=") + videoPath + " latency=0 buffer-mode=none protocols=tcp tcp-timeout=5 timeout=5 ! rtph265depay ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=1 ! h265parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
        // #else
        // if (codeType)
        //     ans = std::string("rtspsrc location=") + videoPath + " latency=0 buffer-mode=none protocols=tcp tcp-timeout=5 timeout=5 ! rtph264depay ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=1 ! h264parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
        // else
        //     ans = std::string("rtspsrc location=") + videoPath + " latency=0 buffer-mode=none protocols=tcp tcp-timeout=5 timeout=5 ! rtph265depay ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=1 ! h265parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=1 drop=true sync=false";
        // #endif

        // #ifdef DECNOTRESIZE
//         if (codeType)
//             ans = std::string("rtspsrc location=") + videoPath + " latency=2000 buffer-mode=auto protocols=tcp tcp-timeout=5 timeout=5 retry=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=5 drop=true sync=false";
//         else
//             ans = std::string("rtspsrc location=") + videoPath + " latency=2000 buffer-mode=auto protocols=tcp tcp-timeout=5 timeout=5 retry=0 ! rtph265depay ! h265parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=5 drop=true sync=false";
// #else
//         if (codeType)
//             ans = std::string("rtspsrc location=") + videoPath + " latency=2000 buffer-mode=auto protocols=tcp tcp-timeout=5 timeout=5 retry=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=5 drop=true sync=false";
//         else
//             ans = std::string("rtspsrc location=") + videoPath + " latency=2000 buffer-mode=auto protocols=tcp tcp-timeout=5 timeout=5 retry=0 ! rtph265depay ! h265parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=5 drop=true sync=false";
// #endif


// #ifdef DECNOTRESIZE
//         if (codeType)
//             ans = std::string("rtspsrc location=") + videoPath + " ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=3 drop=true sync=false";
//         else
//             ans = std::string("rtspsrc location=") + videoPath + " ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=3 drop=true sync=false";
// #else
//         if (codeType)
//             ans = std::string("rtspsrc location=") + videoPath + " ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=3 drop=true sync=false";
//         else
//             ans = std::string("rtspsrc location=") + videoPath + " ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvideoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=3 drop=true sync=false";
// #endif

// // // 路数多要用queue，爱提醒还行，广前有卡住的情况
// #ifdef DECNOTRESIZE
//         if (codeType)
//             ans = std::string("rtspsrc location=") + videoPath + " ! rtph264depay ! h264parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=3 drop=true sync=false";
//         else
//             ans = std::string("rtspsrc location=") + videoPath + " ! rtph265depay ! h265parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=3 drop=true sync=false";
// #else
//         if (codeType)
//             ans = std::string("rtspsrc location=") + videoPath + " ! rtph264depay ! h264parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=3 drop=true sync=false";
//         else
//             ans = std::string("rtspsrc location=") + videoPath + " ! rtph265depay ! h265parse ! nvv4l2decoder ! queue max-size-buffers=1 leaky=1 ! nvvideoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=3 drop=true sync=false";
// #endif

    }
    return ans;
}

class VideoDeal
{
public:
    VideoDeal(std::string videoPath, int gpu_device = 0)
    {
        cudaSetDevice(gpu_device); // 选择gpu卡

        video_path = videoPath;
        std::string codecType = check_codec_type(videoPath);
        if (codecType == "h264")
            videoPath = get_rtsp_name(videoPath, 0, 1280, 720, 1);
        else if (codecType == "hev1" || codecType == "hevc")
            videoPath = get_rtsp_name(videoPath, 0, 1280, 720, 0);
        else
        {
            status = false;
            return;
        }
        // std::lock_guard<std::mutex> lock(mtx_cv); //加不加没有区别，加了有个相机卡住其他相机会开不起来
        cap.open(videoPath, cv::CAP_GSTREAMER); //有时候要很久几十秒才能连上
        bool b = cap.isOpened();
        if (b)
        {
            fps = cap.get(cv::CAP_PROP_FPS);
            if (fps == 0)
            {
                fps = 15;
            }
        }
        status = b;
        std::cout << "[INFO] cap open: " << video_path << "--is: " << status << std::endl;
    }

    ~VideoDeal()
    {
        // std::lock_guard<std::mutex> lock(mtx_cv);
        cap.release();
        std::cout << "[INFO] cap release: " << video_path << std::endl;
    }

    bool getframe(cv::Mat &mat)
    {
        if (!status || !cap.isOpened())
        {
            std::cout << "[ERROR] cap is close: " << video_path << std::endl;
            return false;
        }

        cap.read(mat);

        // if(cams_count>16)
        //     usleep(cams_count*10*1000);//！！！路数多的时候要抽帧————没用。

        // if (video_path.find("rtsp") == std::string::npos)
        //     usleep((1.0 / (float)fps) * 1000000);

        if (mat.empty())
        {
            std::cout << "[ERROR] mat is empty !\n";
            usleep(20000);
            //  return false;
            if (std::time(0) - lastFrameTime > 2)
            {
                std::cout << "[ERROR] lastFrameTime is out !\n";
                return false;
            }
            return true;
        }
        else
        {
            lastFrameTime = std::time(0);
        }
        return true;
    }

    int alarFrameCnt = 0;
    std::string errMsg = "";

    int fps;
    std::string video_path;
    bool status = false;

    int cams_count = 0; // 全局摄像头数量

private:
    cv::VideoCapture cap;
    time_t lastFrameTime;
};
