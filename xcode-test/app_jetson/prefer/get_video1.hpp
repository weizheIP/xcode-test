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

static std::string get_rtsp_name(std::string videoPath, int latency, int width, int height, int codeType)
{
    std::string ans;
    if (videoPath.compare(0, 4, "rtsp") == 0)
    {

#ifdef DECNOTRESIZE
        if (codeType)
            ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp ! rtpjitterbuffer ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=10 drop=true";
        else
            ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp ! rtpjitterbuffer ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! appsink max-buffers=10 drop=true";
#else
        if (codeType)
            ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp !  rtpjitterbuffer ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=10 drop=true";
        else
            ans = std::string("rtspsrc location=") + videoPath + " protocols=tcp !  rtpjitterbuffer ! rtph265depay ! h265parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720 ! videoconvert ! appsink max-buffers=10 drop=true";
#endif


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
        // std::lock_guard<std::mutex> lock(mtx_cv);
        cap.open(videoPath, cv::CAP_GSTREAMER);
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

        if (video_path.find("rtsp") == std::string::npos)
            usleep((1.0 / (float)fps) * 1000000);

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

private:
    cv::VideoCapture cap;
    time_t lastFrameTime;
};
