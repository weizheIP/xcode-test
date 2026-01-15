

#ifndef CDS_SERVER_H
#define CDS_SERVER_H

#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "post/model_base.h"
#include "post_util.h"

// #include "get_video.hpp"
// #include "prefer/get_video1.hpp"

#include "base64.h"
#include "json.h"
#include "common/http/crow_all.h"

class CDS_SERVER
{
public:
    CDS_SERVER(std::string algsName, int serverPort)
    {
        port = serverPort;
        _algsName = algsName;
        s_th = thread(&CDS_SERVER::th_fun, this);
        cout << "[INFO] " << _algsName << " cdsServer init success !" << "\n";
    }
    ~CDS_SERVER()
    {
        s_th.join();
        cout << "[INFO] cdsServer release success !" << "\n";
    }

private:
    void get_frame(std::string rtsp, BlockingQueue<cv::Mat> &frame_queue)
    {
        cv::VideoCapture cap(rtsp);
        if (!cap.isOpened())
        {
            cap.release();
            frame_queue.Push(cv::Mat());
        }

        int ii = 0;
        while (cap.isOpened())
        {
            cv::Mat array;
            cap >> array;
            if (array.empty())
            {
                cout << rtsp << "[ERROR] http cam img empty------------ " << std::endl;
                frame_queue.Push(array);
                break;
            }
            if (ii % 7 == 0)
            {
                int hh = array.rows * 1280 / array.cols;
                cv::resize(array, array, cv::Size(1280, hh));
                frame_queue.Push(array);
            }
            if (ii > (NUM - 1) * 7)
            {
                break;
            }
            ii++;
        }
        cap.release();
        std::cout << "[INFO] http getframe th out!" << "\n";
    }

    void th_fun()
    {
        crow::SimpleApp app;

        detYOLO yolo_model;
        yolo_model.init(_algsName + "/m_enc.pt", 2);

        string serverPath;
        if (_algsName == "crowd_density_statistics")
            serverPath = "crowd_density/";
        else
            serverPath = _algsName + "/";
        serverPath = "/cv/" + serverPath;
        constexpr const char str[100] = {0};
        strcpy((char *)str, serverPath.c_str());

        CROW_ROUTE(app, str)
            .methods("POST"_method)([&](const crow::request &req)
                                    {
                                      
            std::string s;
            std::string rtsp;
            std::thread *t = nullptr;
            try
            {
                Json::Reader reader;
                Json::Value value;
                reader.parse(req.body, value);
                std::cout << "[INFO] http req body:" << req.body << "\n";

                Json::Value out;
                std::vector<cv::Mat> out_frames;
                vector<vector<BoxInfo>> boxss;
                vector<int> counts;                                           
                timeval tv3,  tv3_1;
                gettimeofday(&tv3, NULL);
                // 1. 请求: roi,score,rtsp(ROI是基于preview的1280画的)
                std::vector<std::vector<int>> roi_tmp;
                for (int i = 0; i < value["roi"].size(); i++)
                {
                    std::vector<int> tmp;
                    tmp.push_back(value["roi"][i][0].asInt());
                    tmp.push_back(value["roi"][i][1].asInt());
                    roi_tmp.push_back(tmp);
                }
                double score = value["score"].asDouble();
                std::vector<cv::Point> roi;
                for (int i = 0; i < roi_tmp.size(); i = i + 1)
                {
                    roi.push_back(cv::Point(roi_tmp[i][0], roi_tmp[i][1]));
                }
                rtsp = value["rtsp"].asString();

                //2. 7张图
                BlockingQueue<cv::Mat> frame_queue(10);
                t = new std::thread(&CDS_SERVER::get_frame, this, rtsp, ref(frame_queue));                   
                                                                    
                for (auto j = 0; j < NUM; j++)
                {
                    cv::Mat img;                                              
                    if (frame_queue.Pop(img) != 0)
                    {
                        usleep(1000);
                        continue;
                    }
                    if (img.empty())
                    {
                        std::cout << "http mat is empty! danger!" << std::endl;
                        t->join();
                        delete t;
                        return crow::response("http video open error.");
                    }
                    out_frames.push_back(img);
                    // 3.推理
                    std::vector<BoxInfo> boxes;
                    boxes = yolo_model.infer(img);
                    std::vector<BoxInfo> boxes2;
                    for (auto item : boxes)
                    {
                        if (item.label != 0)
                            continue;
                        if (item.score < score)
                            continue;
                        if (roi.size() != 0){
                            if( cv::pointPolygonTest(roi, cv::Point(int((item.x1 + item.x2) / 2), int((item.y1 + item.y2) / 2)), 1)<=0)
                            continue;
                        }
                        boxes2.push_back(item);
                    }
                    boxss.push_back(boxes2);
                    counts.push_back(boxes2.size());
                }

                // 4.取众数
                int count = int(mode_list(counts));
                        // int index = std::distance(std::begin(counts),count); 
                int indexpos = 0;
                for (int i = 0; i < counts.size(); i++)
                {
                    if (counts[i] == count)
                    {
                        indexpos = i;
                        break;
                    }
                }


                std::string img64;
                std::string drawImage64;

                if (out_frames[indexpos].empty())
                {
                    std::cout << "imencode mat is empty! danger!" << std::endl;
                }
                cv_encode_base64(out_frames[indexpos], img64); 
                for (int i = 0; i < boxss[indexpos].size(); i++)
                {
                    cv::Point p1(boxss[indexpos][i].x1, boxss[indexpos][i].y1);
                    cv::Point p2(boxss[indexpos][i].x2, boxss[indexpos][i].y2);
                    cv::rectangle(out_frames[indexpos], p1, p2, cv::Scalar(0, 255, 0), 5); // bgr
                }
                cv::polylines(out_frames[indexpos], roi, true, cv::Scalar(255, 0, 0), 8);
                cv_encode_base64(out_frames[indexpos], drawImage64); // 20

                Json::Value jwarn;
                for (auto item : boxss[indexpos])
                {
                    Json::Value jboxone;
                    jboxone.append(item.x1);
                    jboxone.append(item.y1);
                    jboxone.append(item.x2);
                    jboxone.append(item.y2);
                    Json::Value jboxone2;
                    jboxone2["boxes"] = jboxone;
                    jboxone2["score"] = item.score;
                    jboxone2["label"] = item.label;
                    jboxone2["dur_time"] = 0;
                    jwarn.append(jboxone2);
                }

                Json::Value j;
                j["type"] = _algsName;
                j["cam"] = rtsp;
                j["time"] = (int)std::time(0);
                j["alarm"] = 1;
                j["size"].append(out_frames[indexpos].cols);
                j["size"].append(out_frames[indexpos].rows);
                if (jwarn.isNull())
                    j["warn"].resize(0);
                else
                    j["warn"] = jwarn;
                out["data"] = j;
                s = out.toStyledString();
                s.pop_back();
                s.pop_back();
                s.pop_back();
                s.pop_back();
                s.append(",\"img\":\"");
                s.append(img64);
                s.append("\",\"drawImg\":\"");
                s.append(drawImage64);
                s.append("\"},\"event\":\"pushVideo\"}");
                std::cout << "[INFO] http num is " << count << "," << rtsp <<"\n";
                frame_queue.Clear();
                t->join();
                delete t;
                
            }
            catch (...)
            {
                std::cout << "[ERROR] " << "," << rtsp << "\n";
                if (t != nullptr)
                {
                    t->join();
                    delete t;
                }
                return crow::response("error");
            }
            return crow::response{s}; });

        // app.port(port).concurrency(2).run(); //58081
        app.port(port).run();
    }

    std::string _algsName;
    int port;
    int NUM = 7;

    std::thread s_th;
};

#endif
