#pragma once

// #include "crow.h"
#include "common/http/crow_all.h"
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "base64.h"
#include "json.h"

// #include "json.hpp"
// using json = nlohmann::json;

extern std::map<std::string, cv::Mat> camImage;
extern std::mutex mtx_camImage;

// 视频解码线程
class HttpVideoGetImage
{
public:
    HttpVideoGetImage()
    {
        std::cout << "[INFO]: start http get image...  " << "\n";
        thread = std::thread(&HttpVideoGetImage::decode, this);
    }

    ~HttpVideoGetImage()
    {
        stop = true;
        thread.join();
    }

    void decode()
    {
        crow::SimpleApp app;

        CROW_ROUTE(app, "/cv/get_rtsp_imgs")
            .methods("POST"_method)([](const crow::request &req)
                                    {
                                        try
                                        {
                                            // auto jj = json::parse(req.body);
                                            // vector<std::string> roi_tmp = jj;

                                            Json::Reader reader;
                                            Json::Value value;
                                            reader.parse(req.body, value);
                                            std::cout << "http request : " << req.body << "\n";

                                            Json::Value out,out_info;
                                            std::string errorMsg="success. ";
                                            int status_code=200;

                                            cv::Mat img;

                                            for (int i = 0; i < value.size(); i++)
                                            {
                                                std::string rtsp_path = value[i].asString();
                                                

                                                {//至强读不到这个变量camImage，61可以；改成debug模式就可以了
                                                    std::lock_guard<std::mutex> lock(mtx_camImage);
                                                    img=camImage[rtsp_path];

                                                    for (const auto& pair : camImage) {
                                                        std::cout << "http cam: " << pair.first<<"-"<< pair.second.size() ;
                                                    }
                                                    std::cout << "--- http end. "<< std::endl;
                
                                                }
                                                if (img.empty()){//重新打开相机取图

                                                                    cv::VideoCapture cap(rtsp_path,cv::CAP_FFMPEG);
                                                                    if (cap.isOpened())
                                                                    {
                                                                        cap.read(img);
                                                                        if (img.empty()){
                                                                            out_info.append("");
                                                                            errorMsg="http img empty. ";
                                                                            status_code=500;
                                                                            std::cout << "http img empty. "  << "\n";
                                                                        }
                                                                        else
                                                                        {
                                                                            std::vector<unsigned char> buf;
                                                                            int w = 1280, h = 0;
                                                                            h = img.rows * 1280 /img.cols;
                                                                            if(h%2)
                                                                                h++;
                                                                            cv::resize(img,img,cv::Size(w,h));
                                                                            cv::imencode(".jpg", img, buf);
                                                                            std::string img64 = base64_encode(&(buf[0]), buf.size());
                                                                            out_info.append(img64);
                                                                            std::cout << "http img cap capture ok. "<<rtsp_path  << "\n";
                                                                        }
                                                                    }
                                                                    else
                                                                    {
                                                                        out_info.append("");
                                                                        errorMsg="http cap not open. ";
                                                                        status_code=500;
                                                                        std::cout << "http cap not open. "  << "\n";                                                    
                                                                    }
                                                                    cap.release();

                                                }else{//一次解码的图片
                                                    std::vector<unsigned char> buf;
                                                    // int w = 1280, h = 0;
                                                    // h = img.rows * 1280 /img.cols;
                                                    // if(h%2)
                                                    //     h++;
                                                    // cv::resize(img,img,cv::Size(w,h));
                                                    auto start = std::chrono::steady_clock::now();

                                                    cv::imencode(".jpg", img, buf);
                                                    std::string img64 = base64_encode(&(buf[0]), buf.size());
                                                    out_info.append(img64);
                                                    std::cout << "http img decode ok. "<<rtsp_path  << "\n";

                                                    auto end = std::chrono::steady_clock::now();
                                                    std::chrono::duration<double> elapsed_seconds = end - start;
                                                    cout << "http time : "  <<  elapsed_seconds.count()<< endl;

                                                }


                                            }
                                            
                                            out["result"]= out_info;
                                            out["status"] = status_code;
                                            out["errorMsg"] = errorMsg;
                                            std::string s = out.toStyledString();
                                            std::cout << "http response : " << errorMsg << "\n";

                                            return crow::response{s};
                                        }
                                        catch (...)
                                        {
                                            std::cout << "[ERROR]: Http get_rtsp_imgs error...  " << "\n";
                                            return crow::response("Http get_rtsp_imgs error");
                                        } });

        // app.port(18080)
        //     .multithreaded()
        //     .run();
        app.port(18080).run();
    }

private:
    std::thread thread;
    std::mutex mtx;
    std::atomic_bool stop{false};
};

// int main()
// {
//     HttpVideoGetImage getImg;
// }

// curl -H "Content-Type: application/json" -X POST -d '["rtsp://admin:HuaWei123@192.168.18.163:554/LiveMedia/ch1/Media1"]' "http://192.168.18.61:18080/cv/get_rtsp_imgs"
// curl -H "Content-Type: application/json" -X POST -d '["rtsp://admin:zmxx116116%2B@192.168.10.78:554/Streaming/Channels/101","rtsp://admin:zmxx116116%2B@192.168.10.77:554/Streaming/Channels/101"]' "http://127.0.0.1:18080/cv/get_rtsp_imgs"
// curl -H "Content-Type: application/json" -X POST -d '["rtsp://admin:zmxx116116%2B@192.168.10.105:554/Streaming/Channels/101"]' "http://127.0.0.1:18080/cv/get_rtsp_imgs"
