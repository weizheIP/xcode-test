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
#include "post/queue.h"

static std::mutex mtx_cv;

class VideoDeal
{
public:
    VideoDeal(std::string videoPath, int gpu_device = 0)
    {

        video_path = videoPath;      
        thread = std::thread(&VideoDeal::decode, this);
        sleep(2);
        std::cout << "[INFO] cap open: " << video_path << "--is: " << status << std::endl;
    }

    ~VideoDeal()
    {
        stop = true;
        thread.join();
        std::cout << "[INFO] cap release: " << video_path << std::endl;
    }
  void decode()
  {
    while (!stop)
    {
      try
      {
        cv::VideoCapture cap(video_path);
        //  cap.set(cv::CAP_PROP_BUFFERSIZE,0);
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
        while (!stop){
            cv::Mat mat;
            cap.read(mat);
            g_images.Push(mat);
            if (video_path.find("rtsp") == std::string::npos)
                usleep((1.0 / (float)fps) * 1000000);
        }
      }
    catch (...)
      {
        std::cout << "[ERROR] cap is close: " << video_path << std::endl;
      }
    }
  }

    bool getframe(cv::Mat &mat)
    {
        cv::Mat tempImg1;
        if (g_images.IsEmpty())
        {
            usleep(20000);
            return false;
        }
        if (g_images.Pop(tempImg1) != 0)
        {
            usleep(10000);
            return false;
        }
        if (tempImg1.empty())
        {
            std::cout << "[ERROR] mat is empty !\n";
            usleep(20000);            
            return false;            
        }
       
        if(tempImg1.cols>=tempImg1.rows){
            if (tempImg1.cols > 1280){
                int hh = tempImg1.rows * 1280 / tempImg1.cols;
                if(hh%2)
                    hh++;
                cv::resize(tempImg1, mat, cv::Size(1280, hh));
            }else{
                mat=tempImg1;
            }
        }else{
            if (tempImg1.rows > 1280){
                int ww = tempImg1.cols * 1280 / tempImg1.rows;
                if(ww%2)
                    ww++;
                cv::resize(tempImg1, mat, cv::Size(ww, 1280));
            }else{
                mat=tempImg1;
            }
        }
        return true;
    }

    int alarFrameCnt = 0;
    std::string errMsg = "";

    int fps=15;
    std::string video_path;
    bool status = false;

private:
    time_t lastFrameTime;
    std::thread thread;
    std::atomic_bool stop{false};
    BlockingQueue<cv::Mat> g_images;
};
