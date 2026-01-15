#pragma once

#define CPPHTTPLIB_THREAD_POOL_COUNT 1 // 默认是cpu核心数太大
#include "httplib.h"

#include "json.hpp"
#include "base64.h"
#include "post/queue.h"
#include "post/model_base.h"
#include "post/base.h"
#include "post/base.h"
#include "license.h"
#include "post_custom.h"

using namespace std;
using namespace httplib;
using json = nlohmann::json;

// std::string sTypeName = "pointer_instrument_detection";

static json M_Process(cv::Mat img, vector<BoxInfo> boxes, json cam_config)
{
    string rtsp_path = cam_config["rtsp"];
    vector<BoxInfo> detectResults = boxes;
    cv::Mat frame = img;
    cv::Mat out_frame = frame.clone();
    int alarm = 0;
    json jresult;
    json jwarn;

    // 1.ROI和置信度
    vector<BoxInfo> generate_boxes;

    // 2.类别过滤
    float conf = cam_config["config"]["score"];
    int limArea = 0;
    if (cam_config["config"].contains("area"))
        limArea = cam_config["config"]["area"];
    for (auto x : detectResults)
    {
        if (conf > x.score)
            continue;
        if ((x.x2 - x.x1) * (x.y2 - x.y1) < (limArea * limArea))
        {
            continue;
        }
        generate_boxes.push_back(x);
    }

    // // 4 时间间隔 6 持续时间 删除
    // 5.工作时间
    if (statusInTimes(cam_config["config"]["work_time"]))
    {
        bool NumEnough = false;

        vector<BoxInfo> output_big;
        vector<float> out_value = calMechanical(generate_boxes, output_big, cam_config["roi"], cam_config["config"]["conf_roi"]);
        // for (int i = 0; i < out_value.size(); i++)
        // {
        //     if (out_value[i] > (float)cam_config["config"]["conf_roi"][i]["Alarm_Value"]) //ct上判断，可以支持多个阈值
        //     {
        //         NumEnough = true;
        //     }
        // }
        // if (NumEnough)
        // {
            // alarm = 1;
        // }
        // 输出
        int i = 0;
        for (auto item : output_big)
        {
            if (item.x1 == 0)
                continue;
            json jboxone = {item.x1, item.y1, item.x2, item.y2};
            json jboxone2;
            jboxone2["boxes"] = jboxone;
            jboxone2["score"] = out_value[i];
            jboxone2["label"] = 0;            
            // if (out_value[i] > (float)cam_config["config"]["conf_roi"][i]["Alarm_Value"])            
                // jwarn.emplace_back(jboxone2);            
            // else            
                jresult.emplace_back(jboxone2);            
            i++;
        }        
    }

    // 4.输出
    std::string img64;
    std::string drawImage64;
    cv_encode_base64(out_frame, img64);
    json out;
    out["type"] = "pointer_instrument_detection2";
    out["cam"] = rtsp_path;
    out["time"] = (int)time(0);
    out["alarm"] = alarm;
    out["size"] = {out_frame.cols, out_frame.rows};
    out["img"] = img64;
    out["drawImg"] = drawImage64;
    out["result"] = jresult;
    out["warn"] = jwarn;
    out["status"] = 0;
    out["errorMsg"] = "Success";

    std::cout << "[INFO] http out is " << alarm << "," << rtsp_path << "\n";
    return out;
}

// 视频解码线程
class HttpMechanicalImage
{
public:
    HttpMechanicalImage()
    {
        std::cout << "[INFO]: start http get image...  " << "\n";
        thread = std::thread(&HttpMechanicalImage::run, this);
    }

    ~HttpMechanicalImage()
    {
        thread.join();
    }
    void run()
    {
        // HTTP
        httplib::Server svr;
        detYOLO yolo_model;
        yolo_model.init("pointer_instrument_detection2/m_enc.pt");
        cout << "[INFO] http start. " << "\n";

        svr.Post("/cv/mechanical/", [&](const Request &req, Response &res)
                 {     
      json out;
      try
      {
        auto jj = json::parse(req.body);
        cout<<"http dbgz request: "<<req.body<<"\n";
        string imgpath = jj["img"];
                
        cv::Mat array; 
        array = cv::imread(imgpath);
        if (array.empty())
        {
            cout << "[ERROR] http img empty------------ " << std::endl;
            json out = {
                {"errorMsg", "open cam error."},
                {"result", ""},
                {"status", 1161},
            };
        }
        // int hh = array.rows * 1280 / array.cols;
        // cv::resize(array, array, cv::Size(1280, hh));
        // 3.推理
        std::vector<BoxInfo> boxes;
        boxes = yolo_model.infer(array);
        out = M_Process(array,boxes,jj);

      }
      catch (...)
      {
        out = {
            {"errorMsg", "error"},
            {"result", ""},
            {"status", 1160},
        };
      }
      auto body = out.dump();
      res.set_content(body, "application/json"); });

        svr.listen("0.0.0.0", 18083);
    }

private:
    std::thread thread;
    std::mutex mtx;
};

// curl -H "Content-Type: application/json" -X POST -d '{"img":"1.jpg", "roi":[],"rtsp":"rtsp://admin:HuaWei123@192.168.3.163:554/LiveMedia/ch1/Media1","config":""}' "http://127.0.0.1:18083/cv/mechanical/"
