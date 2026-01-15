#pragma once

#include <time.h>
#include <thread>
#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <pthread.h>
#include <map>
#include <ctime>
#include <memory>
#include <iostream>
#include <string>
#include "common/ConvertImage.h"
#include "post_util.h"
#include "post_custom.h"

#include "common/alarmVideo/alarm_video.h"

#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

extern string ip_port;
struct outDetectData
{
    cv::Mat img;
    std::vector<BoxInfo> boxes;
    json configs;
};

class PostProcessor
{
public:
    PostProcessor(string _alg_name, string _rtsp_path)
    {
        alg_name = _alg_name;
        rtsp_path = _rtsp_path;
        // thread = std::thread(&PostProcessor::process, this);

		//配置报警基础信息
		t_alarmVideoInfo->algo_name = alg_name;
		t_alarmVideoInfo->rtsp_path = rtsp_path;
		t_alarmVideoInfo->alarm_status = NEED_HANDLER;

        // ws
        std::string host = ip_port.substr(0, ip_port.size() - 5);
        auto const port = "9698";
        auto const results = resolver.resolve(host, port);
        auto ep = net::connect(ws.next_layer(), results);
        host += ':' + std::to_string(ep.port());
        ws.set_option(websocket::stream_base::decorator(
            [](websocket::request_type &req)
            {
                req.set(http::field::user_agent,
                        std::string(BOOST_BEAST_VERSION_STRING) +
                            " websocket-client-coro");
            }));
        ws.handshake(host, "/ws");

        mOldBox.x1 = 0;
        mOldBox.x2 = 0;
        // status = true;
        startTime = 0;
        oldNum = 0;
        sort = std::make_shared<SORT>();
    }

    ~PostProcessor()
    {
        try
        {
        	t_alarmVideoInfo->alarm_status = HANDLERED;
            ws.close(websocket::close_code::normal);
        }
        catch (...)
        {
            std::cerr << "[Error] closing WebSocket. " << std::endl;
        }
        // stop = true;
        // thread.join();
    }

    void Preview(cv::Mat frame)
    {
        // send
        std::time_t t1 = std::time(0);
        std::string img64 = m2s.mat2str(frame);
        json j;
        j["type"] = "preview";
        j["cam"] = rtsp_path;
        j["time"] = t1;
        j["size"] = {frame.cols, frame.rows};
        json value;
        value["data"] = j;
        std::string s = value.dump();
        s.pop_back();
        s.pop_back();
        s.append(",\"img\":\"");
        s.append(img64);
        s.append("\"},\"event\":\"pushVideo\"}");
        ws.write(net::buffer(std::string(s)));
    }

    bool illegal_parking_detection(std::string ID, SORT::TrackingBox frameTrackingResult)
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

    void makeSendMsg(cv::Mat &frame, int &alarm, json &jresult, json &jwarn, json roi)
    {
        if (alg_name == "pointer_instrument_detection")
        { // test，要删掉
            if (!jwarn.empty())
            {
                for (const auto &item : jwarn)
                {
                    cv::Point p1((int)item["boxes"][0], (int)item["boxes"][1] - 4);
                    if (alg_name == "pointer_instrument_detection")
                        cv::putText(frame, std::to_string((float)item["score"]), p1, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255));
                }
            }
        }
        std::string img64 = m2s.mat2str(frame);
        std::string drawImage64;
        if (alarm)
        {
            if (!jresult.empty())
            {
                for (const auto &item : jresult)
                {
                    cv::Point p1((int)item["boxes"][0], (int)item["boxes"][1]);
                    cv::Point p2((int)item["boxes"][2], (int)item["boxes"][3]);
                    cv::rectangle(frame, p1, p2, cv::Scalar(0, 255, 0), 2); // 5
                }
            }
            if (!jwarn.empty())
            {
                for (const auto &item : jwarn)
                {
                    cv::Point p1((int)item["boxes"][0], (int)item["boxes"][1]);
                    cv::Point p2((int)item["boxes"][2], (int)item["boxes"][3]);
                    cv::rectangle(frame, p1, p2, cv::Scalar(0, 0, 255), 2);
                }
            }
            if (alg_name == "pointer_instrument_detection" || alg_name == "cv_machine_room_property_detection" || alg_name == "cv_illegal_parking_across_parking_spaces" || alg_name == "cv_charging_gun_notputback"  || alg_name == "cv_meter_reading_detection" || alg_name == "cv_asset_deficiency_detection")
            // if (is_3d_array(roi))//bug
            {
                for (int j = 0; j < roi.size(); j++)
                {
                    std::vector<cv::Point> points;
                    for (int i = 0; i < roi[j].size(); i++)
                    {
                        points.push_back(cv::Point((int)roi[j][i][0], (int)roi[j][i][1]));
                    }
                    if (roi[j].size())
                        cv::polylines(frame, points, true, cv::Scalar(255, 0, 0), 2); // 8
                }
            }
            else // 单roi
            {
                std::vector<cv::Point> points;
                for (int i = 0; i < roi.size(); i++)
                {
                    points.push_back(cv::Point((int)roi[i][0], (int)roi[i][1]));
                }
                if (roi.size())
                    cv::polylines(frame, points, true, cv::Scalar(255, 0, 0), 2);
            }
            drawImage64 = m2s.mat2str(frame);
			
            // 添加报警消息 
            std::ifstream file_check(t_alarmVideoInfo->alarm_file_name);
            if (t_alarmVideoInfo->alarm_status != HANDLERING || !file_check.good()){
                t_alarmVideoInfo->alarm_file_name = AlarmVideoInfo::CreateVideoFileName(rtsp_path, alg_name);
//                t_alarmVideoInfo->alarm_status = NEED_HANDLER;
                AlarmVideoInfo::AddAlarmInfo(t_alarmVideoInfo);
				std::cout << "[INFO]  == create video msg : " << t_alarmVideoInfo->alarm_file_name << std::endl;
            }			
        }

        // send
        std::time_t t1 = std::time(0);
        json j;
        j["type"] = alg_name;
        j["cam"] = rtsp_path;
        j["time"] = t1;
        j["alarm"] = alarm;
        j["size"] = {frame.cols, frame.rows};
        if (jresult.empty())
            jresult = json::array();
        if (jwarn.empty())
            jwarn = json::array();
        j["result"] = jresult;
        j["warn"] = jwarn;
		j["video"] = t_alarmVideoInfo->alarm_file_name;
        json value;
        value["data"] = j;
        std::string s = value.dump();
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
        ws.write(net::buffer(std::string(s)));
    }

    void IdProcess(outDetectData postmsg, json cam_config, int cam_config_open)
    {
        vector<BoxInfo> detectResults = postmsg.boxes;
        cv::Mat frame = postmsg.img;
        int alarm = 0;
        json jresult;
        json jwarn;

        // 1.ROI和置信度
        vector<BoxInfo> generate_boxes;
        RoiFillter(detectResults, cam_config["roi"], alg_name);

        // 2.类别过滤
        float conf = cam_config["config"]["score"];
        int limArea = 0;
        if (cam_config["config"].contains("area"))
            limArea = cam_config["config"]["area"];
        for (auto x : detectResults)
        {
            if (alg_name == "illegal_parking_detection" && x.label != 0)
                continue;
            if (alg_name == "loitering_detection" && x.label != 0)
                continue;
            if (alg_name == "cv_wandering" && x.label != 0)
                continue;
            if (conf > x.score)
                continue;
            generate_boxes.push_back(x);
        }
        // 3 自动过滤，位置和状态
        int beatBrainFilter = cam_config["config"]["auto_filter"]; // 默认-1

        int a_mode = 0;

        std::time_t t = std::time(0);
        if (oldtime < 0) // 初始化
            oldtime = cam_config["config"]["time"];
        if (!(abs(oldtime - (float)cam_config["config"]["time"]) < 0.0000001))
        {
            std::cout << "[INFO] time is reset:" << alg_name << rtsp_path << std::endl;
            oldtime = cam_config["config"]["time"];
            time_start_dict.clear();
            id_time_interval = 0;
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

        if (statusInTimes(cam_config["config"]["work_time"]))
        {
            std::vector<SORT::TrackingBox> frameTrackingResult = sort->Sortx(bbox, fi++);
            if (fi > 10000000)
            {
                fi = 0;
                sort.reset();
                sort = std::make_shared<SORT>();
                time_start_dict.clear();
            }

            bool isIdLimt = false;
            for (auto item : frameTrackingResult)
            {
                if (item.id >= 2147483600)
                    isIdLimt = true;
                std::map<string, std::time_t>::iterator it;
                string id_label = to_string(item.id) + "_" + to_string(item.label);
                it = time_start_dict.find(id_label);
                if (it == time_start_dict.end())
                    // time_start_dict[id_label] = t - 1;//防止一开始就报警
                    time_start_dict[id_label] = t; // 防止一开始就报警
                // 告警
                if (t - time_start_dict[id_label] > ((int)cam_config["config"]["target_continued_time"]))
                {
                    if (alg_name == "illegal_parking_detection")
                    {
                        if (illegal_parking_detection(id_label, item))
                        // if(illegal_parking_detection(id_label,item,conf)) //rk
                        {
                            // if (a_mode == 0) // 手动
                            time_start_dict[id_label] = t + (float)cam_config["config"]["time"] * 60.0;
                            // else
                            //     time_start_dict[id_label] = t + 24 * 60 * 60;
                            alarm = 1;
                            std::cout << "\033[34m" << "\e[1m" << "[Info] alarm:" << alg_name << "," << rtsp_path << "\e[0m" << std::endl;
                        }
                    }
                    else
                    {

                        // if (a_mode == 0) // 手动
                        // {
                        time_start_dict[id_label] = t + (float)cam_config["config"]["time"] * 60.0;
                        bool filter = true;
                        if (beatBrainFilter)
                            filter = !same_place_detect_arr2(oldTrackBoxs, item);

                        if (id_time_interval <= t && filter)
                        {
                            id_time_interval = t + (float)cam_config["config"]["time"] * 60.0;
                            alarm = 1;
                        }
                        // }
                        // else
                        // {
                        //     alarm = 1;
                        //     time_start_dict[id_label] = t + 24 * 60 * 60;
                        // }
                    }
                }

                json jboxone = {item.box.x, item.box.y, item.box.x + item.box.width, item.box.y + item.box.height};
                json jboxone2;
                jboxone2["boxes"] = jboxone;
                jboxone2["score"] = item.score;
                jboxone2["label"] = item.label;
                jboxone2["id"] = item.id;
                jboxone2["dur_time"] = (int)(t - time_start_dict[id_label]); // 持续时间
                if (time_start_dict[id_label] > t)
                {
                    jwarn.emplace_back(jboxone2);
                }
                else
                {
                    jresult.emplace_back(jboxone2);
                }
            }
            if (isIdLimt)
                time_start_dict.clear();
            if (alarm)
                oldTrackBoxs = frameTrackingResult;
        }

        if (cam_config_open == 1 || (cam_config_open == 0 && alarm == 1))
        {
            makeSendMsg(frame, alarm, jresult, jwarn, cam_config["roi"]);
        }
    }

    void NoIdProcess(outDetectData postmsg, json cam_config, int cam_config_open)
    {
        vector<BoxInfo> detectResults = postmsg.boxes;
        cv::Mat frame = postmsg.img;
        int alarm = 0;
        json jresult;
        json jwarn;

        // 1.ROI和置信度
        vector<BoxInfo> generate_boxes, generate_boxes4;
        if(alg_name == "cv_meter_reading_detection" or alg_name == "cv_machine_room_property_detection")
            MultiRoiFillter(detectResults, cam_config["roi"], alg_name);
        else if(alg_name == "cv_charging_gun_notputback")
            ;
        else
            RoiFillter(detectResults, cam_config["roi"], alg_name);   

        // 2.类别过滤
        float conf = cam_config["config"]["score"];
        int limArea = 0;
        if (cam_config["config"].contains("area"))
            limArea = cam_config["config"]["area"];
        std::vector<int> FilterList;
        if (cam_config["config"].contains("color"))
        {
            for (auto Color_ : cam_config["config"]["color"])
            {
                FilterList.push_back(mCOLORList[Color_]);
            }
        }
        for (auto x : detectResults)
        {
            if (alg_name == "climbing_recognition" && x.label != 0)
                continue;
            if (alg_name == "over_the_wall_detection" && x.label != 0)
                continue;
            if (alg_name == "lane_pedestrian_detection" && x.label != 0)
                continue;
            if (alg_name == "intrusion_detection" && x.label != 0)
                continue;
            if (alg_name == "cv_perimeter_invasion" && x.label != 0)
                continue;
            if (alg_name == "fire_detection" && x.label != 0)
                continue;
            if (alg_name == "mask_detection" && x.label != 0)
                continue;
            if (alg_name == "illegal_parking_detection" && x.label != 0)
                continue;
            if (alg_name == "helmet_detection" && x.label != 1)
                continue;
            if (alg_name == "reflective_clothing_detection" && x.label != 1)
                continue;
            if (alg_name == "chef_hat detection" && x.label != 1)
                continue;
            if (alg_name == "smoke_detection" && x.label != 1)
                continue;
            if (alg_name == "crowd_density_statistics" && x.label != 1)
                continue;
            if (alg_name == "work_clothes_detection" && std::count(FilterList.begin(), FilterList.end(), x.label) != 0) // 允许通行的要过滤
                continue;
            if (alg_name == "gzyssb" && std::count(FilterList.begin(), FilterList.end(), x.label) != 0)
                continue;
            if (alg_name == "over_the_wall_detection" && x.label != 0)
                continue;
            if (alg_name == "road_ponding" && ((x.x2 - x.x1) * (x.y2 - x.y1) < 30 * 30))
                continue;
            if (alg_name == "cv_goggles" && x.label != 2)
                continue;
            // //
            // if ((alg_name == "student_absent_detection" || alg_name == "leaving_post" || alg_name == "cv_site_personnel_limit") && x.label != 1)
            //     continue;
            // if (alg_name == "sleeping_post" && x.label != 0)
            //     continue;
            // if (alg_name == "people_gathering_detection" && x.label != 0)
            //     continue;
            // if ((alg_name == "cv_double_banknote_detection") && x.label != 0)
            //     continue;
            // //
            if (alg_name == "trash_can_full" && (x.label != 1)) // 4个类 0桶1满2垃圾3箱
                continue;
            if (alg_name == "phone_detection" && x.label != 0) // 3
                continue;
            if (alg_name == "play_phone_detection" && x.label != 0) // 3
                continue;
            if (alg_name == "charging_pile_occupied_by_oil_car" && x.label != 1) // 3
                continue;
            if (alg_name == "smoking_detection" && x.label != 0) // 3 0-吸烟/手机 1-人 2头
                continue;
            if (alg_name == "garbage_exposure" && (x.label != 2)) // 6 0123-4-5
                continue;
            if (alg_name == "cv_large_item_inspection" && (x.label == 0 || x.label == 1 || x.label >= 4)) // 6
                continue;
            if (alg_name == "cv_aisle_cargo_detection" && (x.label == 0 || x.label == 1 || x.label >= 4)) // 6
                continue;                
            if (alg_name == "cv_school_uniforms_detection" && (x.label == 0))
                continue;
            if (alg_name == "cv_gas_mask" && (x.label == 0))
                continue;
            if (alg_name == "wash_hands_detection" && (x.label != 0))
                continue;           
            // if (alg_name == "tv_drop_detection" && (x.label != 62))//coco
            //     continue;
            if (conf > x.score)
                continue;
            if ((x.x2 - x.x1) * (x.y2 - x.y1) < (limArea * limArea))
            {
                continue;
            }
            generate_boxes.push_back(x);
        }

        // 3 自动过滤，位置和状态
        int beatBrainFilter = cam_config["config"]["auto_filter"]; // 默认-1

        // 4 时间间隔
        std::time_t t = std::time(0);
        if (oldtime < 0) // 初始化
            oldtime = cam_config["config"]["time"];
        if (!(abs(oldtime - (float)cam_config["config"]["time"]) < 0.0000001))
        {
            std::cout << "--[INFO] time is reset: " << alg_name << rtsp_path << std::endl;
            oldtime = cam_config["config"]["time"];
            time_interval = 0;    // 间隔时间
            time_start_dict1 = t; // 持续时间
            // is_alarmed = 0;
        }
        if (time_start_dict1 == 0)
        {
            time_start_dict1 = t;
            time_interval = t;
        }

        // 5.工作时间
        if (statusInTimes(cam_config["config"]["work_time"]))
        {
            bool NumEnough = false;
            if ((alg_name == "smoking_detection" || alg_name == "phone_detection")) //交集要
            {
                if (smocking_detect2(generate_boxes, detectResults))
                {
                    NumEnough = true;
                }
            }
            else if ((alg_name == "garbage_exposure" || alg_name == "cv_large_item_inspection" || alg_name == "cv_aisle_cargo_detection")) //交集不要
            {
                if ((garbage_detect2(generate_boxes, detectResults, generate_boxes4)) == 1)
                {
                    generate_boxes = generate_boxes4;
                    NumEnough = true;
                }
            }
            else
            {
                if (generate_boxes.size() != 0)
                {
                    NumEnough = true;
                }
            }          

            if (NumEnough)
            {
                // 6 持续时间:  连续3帧没有目标则更新状态时间，一直有目标时间大于持续时间就报警，如果报警后状态没变就不报警
                ntime = t;
                if (t - time_start_dict1 >= (int)cam_config["config"]["target_continued_time"])
                {
                    if (t - time_interval > 0) // 7 间隔时间
                    {
                        // 8 自动位置过滤
                        bool canAlarm = false;
                        bool areaLim = false;
                        if (beatBrainFilter)
                        {
                            for (auto x : generate_boxes)
                            {
                                if (!same_place_detect_arr(oldBoxs, x))
                                {
                                    canAlarm = true;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            canAlarm = true;
                        }
                        // 8 面积过滤
                        if ((alg_name == "fire_and_smoke_detection" ||
                             alg_name == "smoke_detection" ||
                             alg_name == "mouse_detection" ||
                             alg_name == "fire_detection") &&
                            generate_boxes.size() == 1)
                        {
                            if (alg_name == "mouse_detection")
                                limArea = 0;
                            if (no_ID_fire_smoking_detect(mOldBox, generate_boxes[0], limArea))
                            {
                                areaLim = true;
                                std::cout << "[warn] areaLim ...:" << alg_name << "," << rtsp_path << std::endl;
                            }
                        }
                        else
                        {
                            areaLim = true;
                        }

                        if (canAlarm && areaLim)
                        {
                            oldBoxs = generate_boxes;
                            time_interval = t + int((float)cam_config["config"]["time"] * 60.0);
                            // is_alarmed = 1;
                            alarm = 1;
                            time_start_dict1 = time_interval;
                            std::cout << "\033[34m" << "\e[1m" << "[Info] alarm:" << alg_name << "," << rtsp_path << "\e[0m" << std::endl;
                        }
                        else
                        {
                            if (time_start_dict1 - t >= (int)cam_config["config"]["target_continued_time"] && ((time_start_dict1 - t) < ((int)cam_config["config"]["target_continued_time"] + 20)))
                                std::cout << "[warn] same place not alarm1 ...:" << alg_name << "," << rtsp_path << std::endl;
                        }
                    }
                }
            }
            else
            {
                std::time_t nowtime = std::time(0);      
                if (nowtime - ntime > 2)
                {
                    ntime = nowtime;
                    time_start_dict1 = t;
                    n = 0;
                    // is_alarmed = 0;
                    mOldBox.x1 = 0;
                    mOldBox.x2 = 0;
                }
            }

            for (auto item : generate_boxes)
            {
                json jboxone = {item.x1, item.y1, item.x2, item.y2};
                json jboxone2;
                jboxone2["boxes"] = jboxone;
                jboxone2["score"] = item.score;
                jboxone2["label"] = item.label;
                if (t - time_interval > 0)
                    jboxone2["dur_time"] = (int)(t - time_start_dict1); // 持续时间
                else
                    jboxone2["dur_time"] = (int)(t - time_interval); // 持续时间
                if (time_start_dict1 >= t)
                    jwarn.emplace_back(jboxone2);
                else
                    jresult.emplace_back(jboxone2);
            }
        }

        if (cam_config_open == 1 || (cam_config_open == 0 && alarm == 1))
        {
            makeSendMsg(frame, alarm, jresult, jwarn, cam_config["roi"]);
        }
    }

    void WashHandProcess(outDetectData postmsg, json cam_config, int cam_config_open)
    {
        vector<BoxInfo> detectResults = postmsg.boxes;
        cv::Mat frame = postmsg.img;
        int alarm = 0;
        json jresult;
        json jwarn;

        // 1.ROI和置信度
        vector<BoxInfo> generate_boxes, generate_boxes4;
        if(alg_name == "cv_meter_reading_detection" or alg_name == "cv_machine_room_property_detection")
            MultiRoiFillter(detectResults, cam_config["roi"], alg_name);
        else if(alg_name == "cv_charging_gun_notputback")
            ;
        else
            RoiFillter(detectResults, cam_config["roi"], alg_name);            

        // 2.类别过滤
        float conf = cam_config["config"]["score"];
        int limArea = 0;
        if (cam_config["config"].contains("area"))
            limArea = cam_config["config"]["area"];      
        for (auto x : detectResults)
        {            
            if (alg_name == "wash_hands_detection" && (x.label != 0))
                continue;
            if (conf > x.score)
                continue;
            if ((x.x2 - x.x1) * (x.y2 - x.y1) < (limArea * limArea))
            {
                continue;
            }
            generate_boxes.push_back(x);
        }
      
        // 4 时间间隔
        std::time_t t = std::time(0);
        if (oldtime < 0) // 初始化
            oldtime = cam_config["config"]["time"];
        if (!(abs(oldtime - (float)cam_config["config"]["time"]) < 0.0000001))
        {
            std::cout << "--[INFO] time is reset: " << alg_name << rtsp_path << std::endl;
            oldtime = cam_config["config"]["time"];
            time_interval = 0;    // 间隔时间
            time_start_dict1 = t; // 持续时间
            // is_alarmed = 0;
        }
        if (time_start_dict1 == 0)
        {
            time_start_dict1 = t;
            time_interval = t;
        }

        // 5.工作时间
        if (statusInTimes(cam_config["config"]["work_time"]))
        {
           
            if (generate_boxes.size() == 0)
            {
                num_no_obj++;
            }
            else
            {
                jwarn_out.clear();
                for (auto item : generate_boxes)
                {
                    json jboxone = {item.x1, item.y1, item.x2, item.y2};
                    json jboxone2;
                    jboxone2["boxes"] = jboxone;
                    jboxone2["score"] = item.score;
                    jboxone2["label"] = item.label;
                    jboxone2["dur_time"] = (int)(std::time(0)-wash_start_time); // 持续时间                
                    jwarn_out.emplace_back(jboxone2);                    
                }
                wash_frame=frame;
                num_with_obj++;
            }
            if (num_no_obj >= 2 && num_with_obj >= 3 && wash_start_status == 0) //有手
            {
                wash_start_status=  1;
                wash_start_time = std::time(0);
                num_no_obj=0;
                num_with_obj=0;
            }
            if (num_no_obj >= 2  && wash_start_status == 1) //没有手
            {
                if( std::time(0)-wash_start_time < (int)cam_config["config"]["wash_hands_time"])
                {                    
                    alarm = 1;
                    std::cout << "\033[34m" << "\e[1m" << "[Info] alarm:" << alg_name << "," << rtsp_path << "\e[0m" << std::endl; 
                }
                num_with_obj=0;
                wash_start_status = 0;
            }      
           
            for (auto item : generate_boxes)
            {
                json jboxone = {item.x1, item.y1, item.x2, item.y2};
                json jboxone2;
                jboxone2["boxes"] = jboxone;
                jboxone2["score"] = item.score;
                jboxone2["label"] = item.label;
                jboxone2["dur_time"] = (int)(std::time(0)-wash_start_time); // 持续时间                
                if (alarm)
                    jwarn.emplace_back(jboxone2);
                else
                    jresult.emplace_back(jboxone2);
            }
        }

        if (cam_config_open == 1 || (cam_config_open == 0 && alarm == 1))
        {
            if(alarm){
                jwarn=jwarn_out;
                frame=wash_frame;
            }
            makeSendMsg(frame, alarm, jresult, jwarn, cam_config["roi"]);
        }
    }
    void MechanicalProcess(outDetectData postmsg, json cam_config, int cam_config_open)
    {
        vector<BoxInfo> detectResults = postmsg.boxes;
        cv::Mat frame = postmsg.img;
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

        // 4 时间间隔
        std::time_t t = std::time(0);
        if (oldtime < 0) // 初始化
            oldtime = cam_config["config"]["time"];
        if (!(abs(oldtime - (float)cam_config["config"]["time"]) < 0.0000001))
        {
            std::cout << "--[INFO] time is reset: " << alg_name << rtsp_path << std::endl;
            oldtime = cam_config["config"]["time"];
            time_interval = 0;    // 间隔时间
            time_start_dict1 = t; // 持续时间
        }
        if (time_start_dict1 == 0)
        {
            time_start_dict1 = t;
            time_interval = t;
        }

        // 5.工作时间
        if (statusInTimes(cam_config["config"]["work_time"]))
        {
            bool NumEnough = false;

            vector<BoxInfo> output_big;
            vector<float> out_value = calMechanical(generate_boxes, output_big, cam_config["roi"], cam_config["config"]["conf_roi"]);
            for (int i = 0; i < out_value.size(); i++)
            {
                if (out_value[i] > (float)cam_config["config"]["conf_roi"][i]["Alarm_Value"])
                {
                    NumEnough = true;
                }
            }
            if (NumEnough)
            {
                // 6 持续时间:  连续3帧没有目标则更新状态时间，一直有目标时间大于持续时间就报警，如果报警后状态没变就不报警
                ntime = t;
                if (t - time_start_dict1 >= (int)cam_config["config"]["target_continued_time"])
                {
                    if (t - time_interval > 0) // 7 间隔时间
                    {
                        time_interval = t + int((float)cam_config["config"]["time"] * 60.0);
                        alarm = 1;
                        time_start_dict1 = time_interval;
                        std::cout << "\033[34m" << "\e[1m" << "[Info] alarm:" << alg_name << "," << rtsp_path << "\e[0m" << std::endl;
                    }
                }
            }
            else
            {
                std::time_t nowtime = std::time(0);
                if (nowtime - ntime > 2)
                {
                    ntime = nowtime;
                    time_start_dict1 = t;
                }
            }
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
                i++;
                if (t - time_interval > 0)
                    jboxone2["dur_time"] = (int)(t - time_start_dict1); // 持续时间
                else
                    jboxone2["dur_time"] = (int)(t - time_interval); // 持续时间
                if (time_start_dict1 >= t)
                    jwarn.emplace_back(jboxone2);
                else
                    jresult.emplace_back(jboxone2);
            }
        }

        if (cam_config_open == 1 || (cam_config_open == 0 && alarm == 1))
        {
            makeSendMsg(frame, alarm, jresult, jwarn, cam_config["roi"]);
        }
    }

    void EmptyParkingSpace(outDetectData postmsg, json cam_config, int cam_config_open)
    {
        vector<BoxInfo> detectResults = postmsg.boxes;
        cv::Mat frame = postmsg.img;

        int alarm = 0;
        json jresult;
        json jwarn;

        int parking_space_number = cam_config["config"]["parking_space_number"];
        float conf = cam_config["config"]["score"];

        std::time_t t = std::time(0);
        if (oldtime < 0) // 初始化
            oldtime = (float)cam_config["config"]["time"];
        if (!(abs(oldtime - (float)cam_config["config"]["time"]) < 0.0000001))
        {
            std::cout << "--[INFO] time is reset:" << alg_name << rtsp_path << std::endl;
            oldtime = (float)cam_config["config"]["time"];
            time_start_dict1 = t; // 是间隔时间，没有持续时间
        }

        vector<BoxInfo> generate_boxes3;
        for (auto item : detectResults)
        {
            if (conf > item.score)
            {
                continue;
            }
            generate_boxes3.push_back(item);
        }

        int leftCnt;

        if (statusInTimes(cam_config["config"]["work_time"]))
        {
            int carCnt = 0;
            for (auto xx : cam_config["roi"])
            {
                vector<BoxInfo> tmp = generate_boxes3;
                vector<vector<int>> roi = xx;
                RoiFillter(tmp, roi, alg_name);
                carCnt += tmp.size();
            }

            leftCnt = parking_space_number - carCnt;

            if (leftCnt < 0)
                leftCnt = 0;

            if (oldNum != leftCnt || t - time_start_dict1 > oldtime * 60) // 车位数量变化 || 间隔时间达到
            {
                alarm = 1;
            }

            for (auto item : generate_boxes3)
            {
                json jboxone = {item.x1, item.y1, item.x2, item.y2};
                json jboxone2;
                jboxone2["boxes"] = jboxone;
                jboxone2["score"] = item.score;
                jboxone2["label"] = item.label;
                jboxone2["dur_time"] = (int)(t - time_start_dict1); // 持续时间
                jresult.emplace_back(jboxone2);
            }
        }

        if (cam_config_open == 1 || (cam_config_open == 0 && alarm == 1))
        {

            if (t - time_start_dict1 > oldtime * 60)
                time_start_dict1 = t;
            if (oldNum != leftCnt)
                oldNum = leftCnt;

            std::string img64 = m2s.mat2str(frame);

            json j;
            j["type"] = alg_name;
            j["cam"] = rtsp_path;
            j["time"] = (int)t;
            j["alarm"] = alarm;
            j["number"] = oldNum;
            j["size"] = {frame.cols, frame.rows};
            if (jresult.empty())
                jresult = json::array();
            if (jwarn.empty())
                jwarn = json::array();
            j["result"] = jresult;
            j["warn"] = jwarn;
            json value;
            value["data"] = j;
            std::string s = value.dump();
            s.pop_back();
            s.pop_back();
            s.append(",\"img\":\"");
            s.append(img64);
            s.append("\"},\"event\":\"pushVideo\"}");
            ws.write(net::buffer(std::string(s)));
        }
    }

    void GatheringProcess(outDetectData postmsg, json cam_config, int cam_config_open)
    {
        vector<BoxInfo> detectResults = postmsg.boxes;
        cv::Mat frame = postmsg.img;
        int alarm = 0;
        json jresult;
        json jwarn;

        if(alg_name == "cv_asset_deficiency_detection" )
            MultiRoiFillter(detectResults, cam_config["roi"], alg_name);        
        else
            RoiFillter(detectResults, cam_config["roi"], alg_name);

        vector<BoxInfo> generate_boxes;

        float conf = cam_config["config"]["score"];
        for (auto item : detectResults)
        {
            if (conf > item.score)
                continue;
            if(alg_name == "leaving_post" || alg_name == "student_absent_detection"){                
            }else{
                if (item.label != 0)
                    continue;
            }
            // if ( (alg_name == "student_absent_detection"||alg_name == "leaving_post"|| alg_name == "cv_site_personnel_limit") && item.label != 1)
            //     continue;
            // if (alg_name == "sleeping_post" && item.label != 0)
            //     continue;
            // if (alg_name == "people_gathering_detection" && item.label != 0)
            //     continue;
            // if ((alg_name == "cv_double_banknote_detection") && item.label != 0)
            //     continue;
            generate_boxes.push_back(item);
        }

        std::time_t t = std::time(0);
        if (oldtime < 0) // 初始化
            oldtime = (float)cam_config["config"]["time"];
        if (!(abs(oldtime - (float)cam_config["config"]["time"]) < 0.0000001))
        {
            std::cout << "[INFO] time is reset!" << alg_name << rtsp_path << std::endl;
            oldtime = (float)cam_config["config"]["time"];
            time_interval = 0;
            time_start_dict1 = t;
            is_alarmed = 0;
        }
        if (time_start_dict1 == 0)
        {
            time_start_dict1 = t;
            time_interval = t;
            is_alarmed = 0;
        }

        int manCnt = cam_config["config"]["number"];
        if (oldNum != manCnt)
        {
            time_start_dict1 = t;
            oldNum = manCnt;
        }

        int a_mode = cam_config["config"]["auto_filter"];
        if (a_mode != oldMode)
        {
            time_start_dict1 = t;
            oldMode = a_mode;
            time_interval = t;
            is_alarmed = 0;
        }

        if (statusInTimes(cam_config["config"]["work_time"]))
        {
            bool NumEnough = false;
            if (
                ((alg_name == "leaving_post" || alg_name == "student_absent_detection"  || alg_name == "cabinet_indicator_light_detection") && leave_post_detect_(generate_boxes, manCnt)) ||
                // ((alg_name == "leaving_post" ||alg_name == "student_absent_detection") && generate_boxes.size()<manCnt) ||
                (generate_boxes.size() > manCnt && (alg_name == "cv_site_personnel_limit")) ||
                (generate_boxes.size() >= manCnt && (alg_name == "people_gathering_detection" || alg_name == "traffic_jam_detection")) ||
                (generate_boxes.size() >= manCnt && alg_name == "sleeping_post") ||
                (generate_boxes.size() != manCnt && generate_boxes.size() != 0 && (alg_name == "cv_double_banknote_detection")) ||
                (generate_boxes.size() < manCnt && alg_name == "cv_asset_deficiency_detection") 
            )
            {
                NumEnough = true;
                if (t - time_start_dict1 >= (int)cam_config["config"]["target_continued_time"])
                {
                    if (a_mode != 1 && t - time_interval > 0) // mode == 0 手动模式带间隔
                    {
                        time_interval = t + int((float)cam_config["config"]["time"] * 60.0);
                        is_alarmed = 1;
                        alarm = 1;
                        time_start_dict1 = time_interval;
                        std::cout << "\033[34m" << "\e[1m" << "[Info] alarm:" << alg_name << "," << rtsp_path << "\e[0m" << std::endl;
                    }
                    if (a_mode == 1 && is_alarmed == 0 && t - time_interval > 0)
                    { // 自动
                        time_interval = t + int((float)cam_config["config"]["time"] * 60.0);
                        is_alarmed = 1;
                        alarm = 1;
                        time_start_dict1 = time_interval;
                        std::cout << "\033[34m" << "\e[1m" << "[Info] alarm:" << alg_name << "," << rtsp_path << "\e[0m" << std::endl;
                    }
                }
            }
            else
            {
                std::time_t nowtime = std::time(0);
                if (nowtime - ntime >= 2)
                {
                    ntime = nowtime;
                    time_start_dict1 = t;
                    n = 0;
                    is_alarmed = 0;
                }
            }

            for (auto item : generate_boxes)
            {
                if ((alg_name == "leaving_post" || alg_name == "student_absent_detection" ) && item.label == 1)
                    continue;

                json jboxone = {item.x1, item.y1, item.x2, item.y2};
                json jboxone2;
                jboxone2["boxes"] = jboxone;
                jboxone2["score"] = item.score;
                jboxone2["label"] = item.label;
                // if (t - time_interval > 0)
                //     jboxone2["dur_time"] = (int)(t - time_start_dict1); // 持续时间
                // else
                //     jboxone2["dur_time"] = (int)(t - time_interval); // 持续时间

                // if (time_interval > t)
                //     jwarn.emplace_back(jboxone2);
                // else
                //     jresult.emplace_back(jboxone2);
                if (NumEnough)
                    jboxone2["dur_time"] = (int)(t - time_start_dict1); // 持续时间
                if (a_mode == 1)                                        // 自动
                {
                    if (time_interval > t && NumEnough && is_alarmed == 1)
                        jwarn.emplace_back(jboxone2);
                    else
                        jresult.emplace_back(jboxone2);
                }
                else // 手动
                {
                    if (time_interval > t && NumEnough)
                        jwarn.emplace_back(jboxone2);
                    else
                        jresult.emplace_back(jboxone2);
                }
            }
        }
        if (cam_config_open == 1 || (cam_config_open == 0 && alarm == 1))
        {
            makeSendMsg(frame, alarm, jresult, jwarn, cam_config["roi"]);
        }
    }

    void IllegalParkingProcess(outDetectData postmsg, json cam_config, int cam_config_open)
    {
        // PreProcessInfo ans;
        vector<BoxInfo> detectResults = postmsg.boxes;
        cv::Mat frame = postmsg.img;
        int alarm = 0;
        json jresult;
        json jwarn;

        RoiFillter(detectResults, cam_config["roi"], alg_name);
        vector<BoxInfo> generate_boxes;
        int beatBrainFilter = cam_config["config"]["auto_filter"]; // 默认-1

        std::time_t t = std::time(0);
        if (oldtime < 0) // 初始化
            oldtime = cam_config["config"]["time"];
        if (!(abs(oldtime - (float)cam_config["config"]["time"]) < 0.0000001))
        {
            std::cout << "[INFO] time is reset:" << alg_name << rtsp_path << std::endl;
            oldtime = cam_config["config"]["time"];
            time_start_dict.clear();
        }

        // 类别过滤
        float conf = cam_config["config"]["score"];
        for (auto x : detectResults)
        {
            if (conf > x.score)
                continue;
            if (alg_name == "illegal_parking_detection" && x.label != 0)
                continue;
            generate_boxes.push_back(x);
        }
        // ans.boxes = generate_boxes;
        // ans.config = G_config;

        // PreProcessInfo preinfo = ans;

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

        if (statusInTimes(cam_config["config"]["work_time"]))
        {
            std::vector<SORT::TrackingBox> frameTrackingResult = sort->Sortx(bbox, fi++);
            if (fi > 10000000)
            {
                fi = 0;
                sort.reset();
                sort = std::make_shared<SORT>();
                time_start_dict.clear();
            }

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
                }
                // 告警
                bool testBool = illegal_parking_detection(id_label, item);
                if (testBool)
                {
                    ID_T[id_label] = t;

                    static int oldValuex = 0;
                    if (oldValuex != (int)cam_config["config"]["target_continued_time"])
                    {
                        oldValuex = (int)cam_config["config"]["target_continued_time"];
                    }

                    if (t - time_start_dict[id_label] > ((int)cam_config["config"]["target_continued_time"]))
                    {
                        if (!same_place_detect_arr2(oldTrackBoxs, item))
                        {
                            time_start_dict[id_label] = t + (float)cam_config["config"]["time"] * 60.0;
                            alarm = 1;
                            std::cout << "\033[34m" << "\e[1m" << "[Info] alarm:" << alg_name << "," << rtsp_path << "\e[0m" << std::endl;
                        }
                    }
                }
                else
                {
                    // if(a_mode == 0)
                    // if (a_mode == 1 && (t - ID_T[id_label]) > 2)
                    // {
                    //     ID_T[id_label] = t;
                    //     if (time_start_dict[id_label] <= t)
                    //         time_start_dict[id_label] = t - 1;
                    //     // std::cout<<"ID dispear :"<<id_label<<",value:"<<t -1<<std::endl;
                    // }
                }
                // //车牌识别
                // if(alg_name == "license_plate_detection"){
                //     if(alarm == 1){
                // post("/cv/car_plate_rec")
                // frame

                // cv::Mat img;
                // BoxInfo box;
                // BoxInfo carCodeBox;
                // string code;
                //     }
                // }

                json jboxone = {item.box.x, item.box.y, item.box.x + item.box.width, item.box.y + item.box.height};
                json jboxone2;
                jboxone2["boxes"] = jboxone;
                jboxone2["score"] = item.score;
                jboxone2["label"] = item.label;
                jboxone2["id"] = item.id;
                jboxone2["dur_time"] = (int)(t - time_start_dict[id_label]); // 持续时间
                if (time_start_dict[id_label] >= t)
                    jwarn.emplace_back(jboxone2);
                else
                    jresult.emplace_back(jboxone2);
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
                sort.reset();
                sort = std::make_shared<SORT>();
                time_start_dict.clear();
            }
        }
        else
        {
            n = 0;
        }
        if (cam_config_open == 1 || (cam_config_open == 0 && alarm == 1))
        {
            makeSendMsg(frame, alarm, jresult, jwarn, cam_config["roi"]);
        }
    }

    //     void send_msg() {
    //         while (!stop) {
    //         auto result = inputQueue.pop();
    //         // 处理结果
    //         }
    //     }

    //   BlockingQueue<InferenceResult> inputQueue; //如果ws要放线程的话要注意告警消息要保证发送

private:
    //   std::thread thread;
    //   std::atomic_bool stop {false};

    ImagemConverter m2s;
    string rtsp_path, alg_name;

	//video报警
	std::shared_ptr<T_AlarmInfo> t_alarmVideoInfo = std::make_shared<T_AlarmInfo>();
	
    // 工作时长统计
    time_t startTime = 0;
    time_t endTime = 0;

    // 聚集
    int oldNum = 0;
    int is_alarmed = 0;
    int oldMode = 0;

    // 无ID
    std::vector<BoxInfo> oldBoxs; // 判断位置
    BoxInfo mOldBox;              // 烟火只有一个框
    time_t ntime = 0;             // 自动三帧换成一秒
    int n = 0;
    time_t time_interval = 0;    // 时间间隔
    float oldtime = -1;          // 旧的时间间隔
    time_t time_start_dict1 = 0; // 持续时间：开始时间

    // 违停
    std::map<string, std::vector<SORT::TrackingBox>> ID_FireSmokeListHis;
    std::map<string, time_t> ID_T;

    // 有ID
    std::shared_ptr<SORT> sort;
    int fi = 0;
    std::map<string, std::time_t> time_start_dict; // 持续时间：存储每个ID的起始时间
    time_t id_time_interval = 0;
    std::vector<SORT::TrackingBox> oldTrackBoxs;

    // 洗手
    cv::Mat wash_frame;
    json jwarn_out;
    int wash_start_status = 0;
    time_t wash_start_time = 0;
    int num_no_obj = 0;
    int num_with_obj = 0;

    // ws
    net::io_context ioc;
    tcp::resolver resolver{ioc};
    websocket::stream<tcp::socket> ws{ioc};
};
