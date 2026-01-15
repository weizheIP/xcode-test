#pragma once

#include "common/http/crow_all.h"
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <algorithm>
#include <numeric>
#include <random>
#include "common/ConvertImage.h"
#include "base64.h"
#include "json.hpp"

using json = nlohmann::json;
extern string sTypeName;

// 视频解码线程
class HttpCamStatus
{
public:
    HttpCamStatus()
    {
        std::cout << "[INFO]: start http cam status...  " << "\n";
        thread = std::thread(&HttpCamStatus::run, this);
    }

    ~HttpCamStatus()
    {
        // stop = true;
        thread.join();
    }

    void run()
    {
        crow::SimpleApp app;
        CROW_ROUTE(app, "/cv/get_cam_status")
            .methods("POST"_method)([&](const crow::request &req)
        {
            ImagemConverter m2s;
            json out;
            try
            {
                auto jj = json::parse(req.body);
                cout<<"http dbgz request: "<<req.body<<"\n";
                string rtspUrl = jj["rtsp"];
                int getImg = jj["getImg"];

                cv::VideoCapture cap(rtspUrl, cv::CAP_FFMPEG);
                if (!cap.isOpened())
                {
                    std::cout << "[ERROR] Failed to open RTSP stream."<<rtspUrl << std::endl;
                    cap.release();
                    out = {
                        {"errorMsg", "open cam error."},
                        {"result", ""},
                        {"status", 1161},
                    };
                    auto s = out.dump();
                    return crow::response(s);
                }               
                if(getImg == 0)
                {
                    cap.release();
                    out = {
                        {"errorMsg", ""},
                        {"result", ""},
                        {"status", 0},
                    };
                    auto s = out.dump();
                    return crow::response(s);
                }
                cv::Mat array; 
                cap>> array;
                cap>> array;
                cap>> array;
                if (array.empty())
                {
                    cout << "[ERROR] http img empty------------ " << std::endl;
                    out = {
                        {"errorMsg", "img empty error."},
                        {"result", ""},
                        {"status", 1162},
                    };
                    
                }else
                {
                    string img64 = m2s.mat2str(array);
                    out = {
                        {"errorMsg", ""},
                        {"result", img64},
                        {"status", 0},
                    };
                }                
                cap.release();      
                auto s = out.dump();
                return crow::response(s);
            }
            catch (...)
            {
                out = {
                    {"errorMsg", "error"},
                    {"result", ""},
                    {"status", 1160},
                };                
                std::cout << "[ERROR]: Http  error...  " << "\n";                
            } 
            auto s = out.dump();
            return crow::response(s);
        });        
        app.port(19001).run();
    }

private:
    std::thread thread;
    // std::mutex mtx;
    // std::atomic_bool stop{false};
    
};


// curl -H "Content-Type: application/json" -X POST -d '{"rtsp":"rtsp://admin:HuaWei123@192.168.18.163:554/LiveMedia/ch1/Media1","getImg":0}' "http://127.0.0.1:19001/cv/get_cam_status"
