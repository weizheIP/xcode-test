#include <stdio.h>
#include <vector>
#include "string.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <future>
#include <iomanip>

#include "post_process.h"
#include "post/post_util.h"
// #include "post/post_car.h"

// #include "infer.h"

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

extern std::string sWebSockerUrl;
extern std::mutex g_lock_json;
extern Json::Value g_value;
extern Json::Value g_calc_switch;

const double EPS = 0.0000001;

extern BlockingQueue<message_lpr_img> queue_lpr_pre;
extern std::map<std::string, std::shared_ptr<BlockingQueue<message_lpr>>> queue_lpr_post;
extern std::mutex queue_lpr_post_lock;

using namespace std;

Postprocess_comm::Postprocess_comm(string rtsp_path, string typeName) : rtsp_path(rtsp_path), c_typeName(typeName)
{

    trackPtr_ = new SORT();

    status = true;

    send_list = std::make_shared<BlockingQueue<sendMsg>>(5);

    t_sender = new std::thread(&Postprocess_comm::sendThread, this);
}

Postprocess_comm::~Postprocess_comm()
{
    status = false;

    if (t_sender != nullptr && t_sender->joinable())
    {
        t_sender->join();
        delete t_sender;
    }

    if (trackPtr_)
    {
        delete trackPtr_;
    }
}

std::string Postprocess_comm::makeSendMsg(std::shared_ptr<InferOutputMsg> data, Json::Value jresult, Json::Value roi, Json::Value config, Json::Value jwarn, int alarm, bool isRecing)
{
    get_format_time(recNameBuf);
    time_t t1;
    time(&t1);
    // memset(recNameBuf, 0, 200);
    // snprintf(recNameBuf, 200, "/civi/rec/%d.mp4", t1);

    std::string img64, drawImage64;
    if (data->rawImage.empty())
    {
        std::cout << "imencode mat is empty! danger!" << std::endl;
        return nullptr;
    }
    int dstw, dsth;
    cv::Mat dstimg;
    if (data->rawImage.cols != 1280)
    {
        int w = 1280, h = 0;
        h = data->rawImage.rows * 1280 / data->rawImage.cols;
        if (h % 2)
            h++;
        cv::resize(data->rawImage, dstimg, cv::Size(w, h), cv::INTER_AREA);
        cv_encode_base64(dstimg, img64);
        dstw = w;
        dsth = h;
    }
    else
    {
        dstimg = data->rawImage;
        cv_encode_base64(data->rawImage, img64);
        dstw = data->rawImage.cols;
        dsth = data->rawImage.rows;
    }

    // osd画框
    if (alarm)
    {
        if (!jresult.isNull())
        {
            for (int i = 0; i < jresult.size(); i++)
            {
                cv::Point p1(jresult[i]["boxes"][0].asInt(), jresult[i]["boxes"][1].asInt());
                cv::Point p2(jresult[i]["boxes"][2].asInt(), jresult[i]["boxes"][3].asInt());
                cv::rectangle(dstimg, p1, p2, cv::Scalar(0, 255, 0), 5); // bgr
            }
        }
        if (!jwarn.isNull())
        {
            for (int i = 0; i < jwarn.size(); i++)
            {
                cv::Point p1(jwarn[i]["boxes"][0].asInt(), jwarn[i]["boxes"][1].asInt());
                cv::Point p2(jwarn[i]["boxes"][2].asInt(), jwarn[i]["boxes"][3].asInt());
                cv::rectangle(dstimg, p1, p2, cv::Scalar(0, 0, 255), 5);
            }
        }

        std::vector<cv::Point> points;
        for (int i = 0; i < roi.size(); i++)
        {
            points.push_back(cv::Point((int)roi[i][0].asFloat(), (int)roi[i][1].asFloat()));
        }
        if (roi.size())
            cv::polylines(dstimg, points, true, cv::Scalar(255, 0, 0), 8);
        cv_encode_base64(dstimg, drawImage64); // 20
    }

    // 输出json
    Json::Value j;
    j["type"] = c_typeName;
    j["cam"] = rtsp_path;
    j["time"] = (int)t1;
    j["alarm"] = alarm;
    j["size"].append(dstw);
    j["size"].append(dsth);
    if (jresult.isNull())
        j["result"].resize(0);
    else
        j["result"] = jresult;
    if (jwarn.isNull())
        j["warn"].resize(0);
    else
        j["warn"] = jwarn;

    Json::Value jout;
    jout["data"] = j;
    std::string s;
    s = jout.toStyledString();
    s.pop_back();
    s.pop_back();
    s.pop_back();
    s.pop_back();
    s.append(",\"img\":\"");
    s.append(img64);
    if (alarm)
    {
        s.append("\",\"drawImg\":\"");
        s.append(drawImage64);
    }
    s.append("\"},\"event\":\"pushVideo\"}");
    return s;
}

bool Postprocess_comm::illegal_parking_detection(std::string ID, SORT::TrackingBox frameTrackingResult)
{
    float drift = -1.0, drift2 = -1.0, drift3 = -1.0, drift4 = -1.0;
    if (ID_FireSmokeListHis[ID].size() >= 3)
    {
        drift = (float)std::abs(frameTrackingResult.box.width - ID_FireSmokeListHis[ID][0].box.width) / (float)frameTrackingResult.box.width;
        drift2 = (float)std::abs(frameTrackingResult.box.height - ID_FireSmokeListHis[ID][0].box.height) / (float)frameTrackingResult.box.height;
        drift3 = (float)std::abs(frameTrackingResult.box.x - ID_FireSmokeListHis[ID][0].box.x);
        drift4 = (float)std::abs(frameTrackingResult.box.y - ID_FireSmokeListHis[ID][0].box.y);
        ID_FireSmokeListHis[ID].erase(ID_FireSmokeListHis[ID].begin());
    }

    if (drift > 0.03 || drift2 > 0.03 || drift3 > 5 || drift4 > 5)
    {
        ID_FireSmokeListHis[ID].push_back(frameTrackingResult);
        return false;
    }
    else if (drift < 0 && drift2 < 0 && drift3 < 0 && drift4 < 0)
    {
        ID_FireSmokeListHis[ID].push_back(frameTrackingResult);
        return false;
    }
    else
    {
        ID_FireSmokeListHis[ID].push_back(frameTrackingResult);
        return true;
    }
}

std::string Postprocess_comm::process(std::shared_ptr<InferOutputMsg> data, processInfo &info)
{
    vector<BoxInfo> detectResults = data->result;

    g_lock_json.lock();
    Json::Value Roi = Get_Rtsp_Json(g_value, rtsp_path)["roi"];
    Json::Value G_config = Get_Rtsp_Json(g_value, rtsp_path)["config"];
    Json::Value temp = g_calc_switch;
    g_lock_json.unlock();

    std::vector<cv::Point> contours;
    int previewH = 1280.f / (float)data->rawImage.cols * (float)data->rawImage.rows;
    if (previewH % 2 != 0)
        previewH++;
    for (int i = 0; i < Roi.size(); i++)
    {
        int x = (float)Roi[i][0].asInt() * ((float)data->rawImage.cols / (float)1280.f);
        int y = (float)Roi[i][1].asInt() * ((float)data->rawImage.rows / (float)previewH);
        contours.push_back(cv::Point(x, y));
    }
    RoiFillter_car(detectResults, contours);

    vector<BoxInfo> generate_boxes;

    // int a_mode = 0;

    if (oldtime < 0) // 初始化
        oldtime = G_config["time"].asFloat();
    if (!(abs(oldtime - G_config["time"].asFloat()) < EPS))
    {
        std::cout << "[INFO] time is reset:" << c_typeName << rtsp_path << std::endl;
        oldtime = G_config["time"].asFloat();
        time_start_dict.clear();
    }

    // 类别过滤
    float conf = G_config["score"].asFloat();
    for (auto x : detectResults)
    {
        if (conf > x.score)
            continue;
        if (c_typeName == "illegal_parking_detection" && x.label != 0)
            continue;
        generate_boxes.push_back(x);
    }

    std::vector<std::vector<float>> bbox;
    for (auto item : generate_boxes)
    {
        std::vector<float> obj;
        obj.push_back(item.x1);
        obj.push_back(item.y1);
        obj.push_back(item.x2);
        obj.push_back(item.y2);
        obj.push_back(item.score);
        obj.push_back(item.label);
        bbox.push_back(obj);
    }

    int alarm = 0;
    Json::Value jwarn, jresult;
    if (statusInTimes(G_config["work_time"]))
    {
        std::vector<SORT::TrackingBox> frameTrackingResult = trackPtr_->Sortx(bbox, fi++);
        if (fi > 10000000)
        {
            fi = 0;
            delete trackPtr_;
            trackPtr_ = new SORT();
            time_start_dict.clear();
        }
        std::time_t t = std::time(0);
        struct tm st = {0};
        localtime_r(&t, &st);
        bool isIdLimt = false;
        for (auto item : frameTrackingResult)
        {
            if (item.id >= 2147483600)
                isIdLimt = true;
            std::map<string, std::time_t>::iterator it;
            string id_label = to_string(item.id) + "_" + to_string(item.label);
            it = time_start_dict.find(id_label);
            if (it == time_start_dict.end())
            {
                time_start_dict[id_label] = t; // 防止一开始就报警
                // std::cout<<"NEW ID:"<<id_label<<",value:"<<t<<std::endl;
            }
            // 告警
            // bool testBool = illegal_parking_detection(id_label, item);
            // if (testBool)
            // {
                ID_T[id_label] = t;

                static int oldValuex = 0;
                if (oldValuex != G_config["target_continued_time"].asInt())
                {
                    oldValuex = G_config["target_continued_time"].asInt();
                }

                if (t - time_start_dict[id_label] > (G_config["target_continued_time"].asInt()))
                {
                    if (!same_place_detect_arr2(oldTrackBoxs, item))
                    {
                        time_start_dict[id_label] = t + G_config["time"].asFloat() * 60.0;
                        alarm = 1;
                        std::cout << "\033[34m" << "\e[1m" << "[Info] alarm:" << c_typeName
                                  << ",nowTime:" << st.tm_year + 1900 << "-" << st.tm_mon + 1 << "-" << st.tm_mday << " " << st.tm_hour << ":" << st.tm_min << ":" << st.tm_sec
                                  << "," << rtsp_path << "\e[0m" << std::endl;
                    }
                }
            // }

            // //车牌识别
            BoxInfo carCodeBoxInfo;
            std::string carCode;
            if (c_typeName == "license_plate_detection")
            {
                if (alarm == 1)
                {

                    cv::Mat img = data->rawImage;
                    BoxInfo box = {item.box.x, item.box.y, item.box.x+item.box.width, item.box.y+item.box.height};
					if (box.x1 < 0)
						box.x1 = 0;
					if (box.y1 < 0)
						box.y1 = 0;
					if (box.x2 > img.cols)
						box.x2 = img.cols;
					if (box.y2 > img.rows)
						box.y2 = img.rows;
                    message_lpr_img premsg;
                    premsg.frame = img;
                    premsg.videoIndex = rtsp_path;
                    premsg.inBox = box;
                    if (queue_lpr_pre.Push(premsg, true) != 0)
                    {
                        usleep(1000);
                    }
                    // 要同步，等算完100s
                    message_lpr postmsg;
                    queue_lpr_post_lock.lock();
                    std::shared_ptr<BlockingQueue<message_lpr>> aa = queue_lpr_post[rtsp_path];
                    queue_lpr_post_lock.unlock();
                    if (aa->Pop(postmsg, 100 * 1000) != 0)
                    {
                        usleep(1000);
                        continue;
                    }
                    carCodeBoxInfo = postmsg.detectData;
                    carCode = postmsg.lprCode;
                }
            }
            // Json::Value jboxone;
            // jboxone.append(carCodeBoxInfo.x1 * (float) 1280.f/ ((float) data->rawImage.cols));
            // jboxone.append(carCodeBoxInfo.y1* ((float) previewH / (float) data->rawImage.rows));
            // jboxone.append((carCodeBoxInfo.x2) * ((float) 1280.f/ (float) data->rawImage.cols ));

            Json::Value jboxone;
            // jboxone.append(item.box.x);
            // jboxone.append(item.box.y);
            // jboxone.append(item.box.x + item.box.width);
            // jboxone.append(item.box.y + item.box.height);
            int jx1 = (float)item.box.x * (float)1280.f / ((float)data->rawImage.cols);
            int jy1 = (float)item.box.y * ((float)previewH / (float)data->rawImage.rows);
            int jx2 = (float)(item.box.x + item.box.width) * ((float)1280.f / (float)data->rawImage.cols);
            int jy2 = (float)(item.box.y + item.box.height) * ((float)previewH / (float)data->rawImage.rows);
            jboxone.append(jx1);
            jboxone.append(jy1);
            jboxone.append(jx2);
            jboxone.append(jy2);
            Json::Value jboxone2;
            jboxone2["boxes"] = jboxone; // 车
            jboxone2["score"] = item.score;
            jboxone2["label"] = item.label;
            jboxone2["id"] = item.id;
            jboxone2["dur_time"] = (int)(t - time_start_dict[id_label]); // 持续时间
            jboxone2["car_code"] = carCode;
            if (time_start_dict[id_label] >= t)
                jwarn.append(jboxone2);
            else
                jresult.append(jboxone2);
        }
        if (isIdLimt)
            time_start_dict.clear();
        if (alarm)
            oldTrackBoxs = frameTrackingResult;
    }

    if (generate_boxes.size() == 0)
    {
        n++;
        if (n >= 10000)
        {
            n = 0;
            fi = 0;
            delete trackPtr_;
            trackPtr_ = new SORT();
            time_start_dict.clear();
        }
    }
    else
    {
        n = 0;
    }

    string str1 = (temp)["data"]["rtsp"].asString();
    string str2 = rtsp_path;
    string str3 = (temp)["data"]["alg_id"].asString();
    bool switchIsON = ((str1 == str2) && str3 == c_typeName);
    std::string s;
    if (switchIsON || (!switchIsON && alarm == 1))
    {
        s = makeSendMsg(data, jresult, Roi, G_config, jwarn, alarm, info.isRecing);
        info.recName = recNameBuf;
        info.isAlarm = alarm;
    }

    return s;
}

void Postprocess_comm::sendThread()
{
    sendMsg temp;
    while (this->status)
    {
        try
        {
            timeval t1, t2;
            // ws
            std::string wst = sWebSockerUrl; // ws://127.0.0.1:9698/ws
            std::cout << "ws url:" << sWebSockerUrl << std::endl;
            int index_1 = wst.find("//");
            wst = wst.substr(index_1 + 2, wst.size());
            index_1 = wst.find("/ws");
            wst = wst.substr(0, index_1);

            index_1 = wst.find(":");
            std::string IP = wst.substr(0, index_1);

            auto const port = "9698";
            net::io_context ioc;
            tcp::resolver resolver{ioc};
            websocket::stream<tcp::socket> ws{ioc};
            auto const results = resolver.resolve(IP, port);
            auto ep = net::connect(ws.next_layer(), results);
            IP += ':' + std::to_string(ep.port());
            ws.set_option(websocket::stream_base::decorator(
                [](websocket::request_type &req)
                {
                    req.set(http::field::user_agent,
                            std::string(BOOST_BEAST_VERSION_STRING) +
                                " websocket-client-coro");
                }));
            ws.handshake(IP, "/ws");
            if (temp.msg.size() > 0) // 重连过后重复发送上次失败的数据
            {
                ws.write(net::buffer(std::string(temp.msg)));
            }

            while (this->status)
            {
                if (send_list->GetSize() > 0)
                {
                    if (send_list->Pop(temp) == 0)
                    {
                        if (send_list->GetSize() > 3 && temp.isA == false)
                            continue;
                        ws.write(net::buffer(std::string(temp.msg)));
                    }
                    else
                        usleep(10000);
                }
                else
                    usleep(10000);
            }
            ws.close(websocket::close_code::normal);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[ERROR] ws send error:" << e.what() << '\n';
        }
    }
    send_list->Clear();
    send_list.reset();
    std::cout << "[INFO] send thread down" << std::endl;
}
