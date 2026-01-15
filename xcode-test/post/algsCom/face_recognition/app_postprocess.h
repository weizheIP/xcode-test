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
#include "post/algsCom/post_util.h"
#include "post/algsCom/post_custom.h"
// #include "post/algsCom/face_recognition/face_db.h"
// #include "post/algsCom/face_recognition/face_db.h"

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

// extern float MaxConfLimit;

extern string ip_port;
struct outDetectData
{
    cv::Mat img;
    std::vector<BoxInfo> boxes;
    std::vector<CarBoxInfo> carinfos;
    std::vector<faceInfo> faceinfos;
    json configs;
};

static bool in_roi_check_(std::vector<cv::Point> contours, CarBoxInfo objs, float pro)
{
    std::vector<cv::Point> carContours;
    for (int i = 0; i < 4; i++)
    {
        carContours.push_back(cv::Point(objs.landmark[2 * i], objs.landmark[2 * i + 1]));
    }
    std::vector<cv::Point> convexHull1, convexHull2;
    cv::convexHull(contours, convexHull1);
    cv::convexHull(carContours, convexHull2);
    // 计算交集
    std::vector<cv::Point> intersectionPoints;
    cv::intersectConvexConvex(convexHull1, convexHull2, intersectionPoints);
    if (intersectionPoints.size() == 0)
        return false;
    // 计算交集的面积
    double intersectionArea = cv::contourArea(intersectionPoints);
    // 计算交集面积于车面积的占比
    double proportion = intersectionArea / cv::contourArea(carContours);
    // std::cout<<"[INFO] proportion:"<<proportion<<"\n";

    return proportion > pro ? true : false;
}
static void CarBoxInfo2BoxInfo(vector<CarBoxInfo> &carboxes, vector<BoxInfo> &boxes)
{
    for (auto x : carboxes)
    {
        BoxInfo box;
        box.x1 = x.x1;
        box.y1 = x.y1;
        box.x2 = x.x2;
        box.y2 = x.y2;
        box.score = x.score;
        box.label = x.label;
        boxes.push_back(box);
    }
}

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
        std::string img64;
        std::string drawImage64;

        cv::Mat dstimg;
        if (frame.cols != 1280)
        {
            int w = 1280, h = 0;
            h = frame.rows * 1280 / frame.cols;
            if (h % 2)
                h++;
            cv::resize(frame, dstimg, cv::Size(w, h), cv::INTER_AREA);
            img64 = m2s.mat2str(dstimg);
        }
        else
        {
            dstimg = frame.clone();
            img64 = m2s.mat2str(dstimg);
        }

        if (alarm)
        {
            if (!jresult.empty())
            {
                for (const auto &item : jresult)
                {
                    cv::Point p1((int)item["boxes"][0], (int)item["boxes"][1]);
                    cv::Point p2((int)item["boxes"][2], (int)item["boxes"][3]);
                    cv::rectangle(dstimg, p1, p2, cv::Scalar(0, 255, 0), 2); // 5
                }
            }
            if (!jwarn.empty())
            {
                for (const auto &item : jwarn)
                {
                    cv::Point p1((int)item["boxes"][0], (int)item["boxes"][1]);
                    cv::Point p2((int)item["boxes"][2], (int)item["boxes"][3]);
                    cv::rectangle(dstimg, p1, p2, cv::Scalar(0, 0, 255), 2);
                }

            }
            if (alg_name == "cv_illegal_parking_across_parking_spaces" || alg_name == "cv_charging_gun_notputback"  || \
                alg_name == "cv_meter_reading_detection" ||  alg_name == "illegal_operation_phone_terminal_detection" || alg_name == "anti_theft_detection") // 多roi：跨车位和充电枪
            {
                for (int j = 0; j < roi.size(); j++)
                {
                    std::vector<cv::Point> points;
                    for (int i = 0; i < roi[j].size(); i++)
                    {
                        points.push_back(cv::Point((int)roi[j][i][0], (int)roi[j][i][1]));
                    }
                    if (roi[j].size())
                        cv::polylines(dstimg, points, true, cv::Scalar(255, 0, 0), 2); // 8
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
                    cv::polylines(dstimg, points, true, cv::Scalar(255, 0, 0), 2);
            }
            drawImage64 = m2s.mat2str(dstimg);
			// 添加报警消息 
//			  std::ifstream file_check(t_alarmVideoInfo->alarm_file_name);|| !file_check.good()
			if (AlarmVideoInfo::switchAlarm){
				if (t_alarmVideoInfo->alarm_status != HANDLERING ){
					t_alarmVideoInfo->alarm_file_name = AlarmVideoInfo::CreateVideoFileName(rtsp_path, alg_name);
	//				  t_alarmVideoInfo->alarm_status = NEED_HANDLER;
					AlarmVideoInfo::AddAlarmInfo(t_alarmVideoInfo);
					std::cout << "[INFO]  == create video msg : " << t_alarmVideoInfo->alarm_file_name << std::endl;
				}	
			}else{
				t_alarmVideoInfo->alarm_file_name = "";
			}
        }

        // send
        std::time_t t1 = std::time(0);
        json j;
        j["type"] = alg_name;
        j["cam"] = rtsp_path;
        j["time"] = t1;
        j["alarm"] = alarm;
        j["size"] = {dstimg.cols, dstimg.rows};
        if (jresult.empty())
            jresult = json::array();
        if (jwarn.empty())
            jwarn = json::array();
        j["result"] = jresult;
        j["warn"] = jwarn;
//		if (!jwarn.empty())
		j["video"] = t_alarmVideoInfo->alarm_file_name;
        // j["stranger"] = jstranger;
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
        if (alg_name == "cv_charging_gun_notputback" || alg_name == "cv_meter_reading_detection")
            ;
        else if (alg_name == "illegal_operation_phone_terminal_detection")
        {
            if (cam_config["roi"].size() ==2)            
                RoiFillter(detectResults, cam_config["roi"][0], alg_name);
        }
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
            if ((alg_name == "fire_detection" || alg_name == "meiya_fire_detection") && x.label != 0)
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
            if (alg_name == "cv_school_uniforms_detection" && (x.label == 0))
                continue;
            if (alg_name == "cv_gas_mask" && (x.label == 0))
                continue;
            if (alg_name == "wash_hands_detection" && (x.label != 0))
                continue;           
            // if (alg_name == "tv_drop_detection" && (x.label != 62))//coco
            //     continue;
            if (alg_name == "cv_iden_minors" && x.label == 1)
                continue;
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
            if ((alg_name == "smoking_detection" || alg_name == "phone_detection"))
            {
                if (smocking_detect2(generate_boxes, detectResults))
                {
                    NumEnough = true;
                }
            }
            else if ((alg_name == "garbage_exposure" || alg_name == "cv_large_item_inspection"))
            {
                if ((garbage_detect2(generate_boxes, detectResults, generate_boxes4)) == 1)
                {
                    generate_boxes = generate_boxes4;
                    NumEnough = true;
                }
            }
            else if (alg_name == "illegal_operation_phone_terminal_detection" )
            {
                if (cam_config["roi"].size() ==2){

//					std::cout << "==== cam_config[\"config\"][\"safe_distance\"] : " << cam_config["config"]["safe_distance"] << std::endl;
					int dist_mul = 3; // 默认值
					if (cam_config.contains("config") && cam_config["config"].contains("safe_distance")) {
						try {
							dist_mul = cam_config["config"]["safe_distance"];
						} catch (...) {
							dist_mul = 3; // 类型不对时也回退默认值
						}
					}
                    if (illegal_operation_phone(generate_boxes,cam_config["roi"][1], dist_mul))
                    {
                        NumEnough = true;
                    }
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
            	if (alg_name == "illegal_operation_phone_terminal_detection" && (item.label == 2)) continue; // 过滤手和手臂识别框 item.label == 1 || 
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
        int manCnt;
        if(alg_name == "cv_sales_terminal_out_location")
            manCnt = 0;
        else       
            manCnt = cam_config["config"]["number"];
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
                (generate_boxes.size() < 1 && alg_name == "cv_sales_terminal_out_location") ||
                (generate_boxes.size() != manCnt && generate_boxes.size() != 0 && (alg_name == "cv_double_banknote_detection")))
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

    void AcrossPakingProces(outDetectData postmsg, json cam_config, int cam_config_open)
    {
        vector<CarBoxInfo> detectResults = postmsg.carinfos;
        cv::Mat frame = postmsg.img;
        int alarm = 0;
        json jresult;
        json jwarn;

        float pro = cam_config["config"]["pressure_line_offset"];
        int beatBrainFilter = cam_config["config"]["auto_filter"];
        float conf = cam_config["config"]["score"];

        vector<CarBoxInfo> generate_boxes3;
        for (auto x : detectResults)
        {
            if (conf > x.score)
            {
                continue;
            }
            generate_boxes3.push_back(x);
        }

        if (0)
        {
            for (int i = 0; i < generate_boxes3.size(); i++)
            {

                std::vector<cv::Point> points;
                for (int j = 0; j < 4; j++)
                {
                    points.push_back(cv::Point(generate_boxes3[i].landmark[j * 2], generate_boxes3[i].landmark[j * 2 + 1]));
                }

                cv::polylines(frame, points, true, cv::Scalar(255, 255, 0), 4);
            }
        }

        std::time_t t = std::time(0);
        if (oldtime < 0) // 初始化
            oldtime = cam_config["config"]["time"];
        if (!(abs(oldtime - (float)cam_config["config"]["time"]) < 0.0000001))
        {
            std::cout << "--[INFO] time is reset:" << alg_name << rtsp_path << std::endl;
            oldtime = cam_config["config"]["time"];
            time_interval = 0;
            time_start_dict1 = t;
            is_alarmed = 0;
        }
        if (time_start_dict1 == 0)
        {
            time_start_dict1 = t;
            time_interval = t;
        }

        if (statusInTimes(cam_config["config"]["work_time"]))
        {
            bool isAllAcross = false;
            std::vector<std::vector<cv::Point>> contourss;
            vector<vector<vector<int>>> Roi = cam_config["roi"];
            for (int i = 0; i < Roi.size(); i++)
            {
                std::vector<cv::Point> contours;
                for (int j = 0; j < Roi[i].size(); j++)
                    contours.push_back(cv::Point(Roi[i][j][0], Roi[i][j][1]));
                contourss.push_back(contours);
            }

            if (contourss.size() == 0)
                generate_boxes3.resize(0);

            if (generate_boxes3.size() != 0)
            {
            ReDoFind:

                for (int i = 0; i < generate_boxes3.size(); i++)
                {
                    int crossSize = 0;
                    for (int j = 0; j < contourss.size(); j++)
                    {
                        if (in_roi_check_(contourss[j], generate_boxes3[i], pro))
                        {
                            crossSize++;
                        }
                    }
                    if (crossSize < 2)
                    {
                        generate_boxes3.erase(generate_boxes3.begin() + i);
                        goto ReDoFind;
                    }
                }
            }

            if (generate_boxes3.size() != 0)
            {
                ntime = t;
                if (t - time_start_dict1 >= (int)cam_config["config"]["target_continued_time"])
                {
                    // if (a_mode==0 && t-time_interval>0){ //mode == 0 手动模式带间隔
                    if (t - time_interval > 0)
                    { // mode == 0 手动模式带间隔

                        bool canAlarm = false;
                        std::vector<BoxInfo> generate_boxes4;
                        if (beatBrainFilter)
                        {

                            CarBoxInfo2BoxInfo(generate_boxes3, generate_boxes4);
                            for (auto x : generate_boxes4)
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
                        if (canAlarm)
                        {
                            oldBoxs = generate_boxes4;
                            time_interval = t + int((int)cam_config["config"]["time"] * 60.0);
                            is_alarmed = 1;
                            alarm = 1;
                            time_start_dict1 = time_interval;
                            std::cout << "\033[34m" << "\e[1m" << "[Info] alarm:" << alg_name << "," << rtsp_path << "\e[0m" << std::endl;
                        }
                        else
                        {
                            if (time_start_dict1 - t >= (int)cam_config["config"]["target_continued_time"] && ((time_start_dict1 - t) < ((int)cam_config["config"]["target_continued_time"] + 20)))
                                std::cout << "[warn] same place not alarm1 ...:" << alg_name << "," << rtsp_path << std::endl;
                            ;
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
                    is_alarmed = 0;
                    mOldBox.x1 = 0;
                    mOldBox.x2 = 0;
                }
            }

            for (auto item : generate_boxes3)
            {
                json jboxone = {item.x1, item.y1, item.x2, item.y2};
                json jboxone2;
                jboxone2["boxes"] = jboxone;
                jboxone2["score"] = item.score;
                jboxone2["dur_time"] = (int)(t - time_start_dict1); // 持续时间

                if (time_start_dict1 >= t)
                    jwarn.emplace_back(jboxone2);
                else
                    jresult.emplace_back(jboxone2);
            }
        }
        else
        {
            time_start_dict1 = t;
            is_alarmed = 0;
        }
        if (cam_config_open == 1 || (cam_config_open == 0 && alarm == 1))
        {
            makeSendMsg(frame, alarm, jresult, jwarn, cam_config["roi"]);
        }

        return;
    }
	
	void FaceProcess1(outDetectData postmsg, json cam_config, int cam_config_open)
	{
		vector<faceInfo> ans = postmsg.faceinfos;
		cv::Mat frame = postmsg.img;
		int alarm = 0;
		json jresult;
		json jwarn;

		std::time_t t = std::time(0);
		struct tm st = {0};
		localtime_r(&t, &st);
		int previewH = 1280.f / (float)frame.cols * (float)frame.rows;
		if (previewH % 2 != 0)
			previewH++;

		float score = cam_config["config"]["score"];
		int limArea = cam_config["config"]["area"];
		float conf = (1 - (float)cam_config["config"]["conf"]) * 2.0; // rk 相似度和距离相反

		if (statusInTimes(cam_config["config"]["work_time"]))
		{
			if (oldtime < 0) // 初始化
				oldtime = cam_config["config"]["time"];
		 
			if (!(abs(oldtime - (float)cam_config["config"]["time"]) < 0.0000001))
			{
				std::cout << "--[INFO] time is reset:" << alg_name << rtsp_path << std::endl;
				oldtime = cam_config["config"]["time"];
				time_start_dict1 = t; // 陌生人的持续时间
				time_interval = t;
				faceIdTimeMap.clear(); // 每个人员的持续时间
			}


			if (time_start_dict1 == 0)
			{
				time_start_dict1 = t;
				time_interval = t;
			}

			if (ans.size() > 0)
			{
			reDel:
				for (auto it = faceIdTimeMap.begin(); it != faceIdTimeMap.end(); it++)
				{
					if (t - it->second > (int)cam_config["config"]["target_continued_time"] + 3)
					{
						faceIdTimeMap.erase(it->first);
						goto reDel;
					}
				}
				for (int i = 0; i < ans.size(); i++)
				{
					auto it = faceIdTimeMap.find(ans[i].id);
					int whatPeo = 0;														 // 0 = 数据库里的人，1 = 陌生人，2 = 不达标人
					if (ans[i].id != -2 && it == faceIdTimeMap.end() && conf >= ans[i].conf) // 熟人还没计时（id为-2代表数据库没有这个人）
					{
						if (ans[i].id != -1) //-1是要删除的人
						{
							whatPeo = 0;
							faceIdTimeMap[ans[i].id] = t;
						}
						else
						{
							whatPeo = 2;
						}
					}
					else
					{
						if (1.6 < ans[i].conf || ans[i].id == -2) // 陌生人：1.6对应的相似度是0.2，小于0.2为陌生人
						{
							whatPeo = 1; // 陌生人
							ans[i].fileName = "";
							ntime_1 = t;
							if (t - time_start_dict1 > (int)cam_config["config"]["target_continued_time"] && t - time_interval > (int)cam_config["config"]["target_continued_time"])
							{
								time_start_dict1 = (time_t)(t + (oldtime * 60.0));
								if (alg_name != "personnel_duty_detection"){
									alarm = 1;
									std::cout << "\033[34m" << "\e[1m" << "[Info] alarm:" << alg_name
											<< ",nowTime:" << st.tm_year + 1900 << "-" << st.tm_mon + 1 << "-" << st.tm_mday << " " << st.tm_hour << ":" << st.tm_min << ":" << st.tm_sec
											<< "," << rtsp_path << "\e[0m" << std::endl;
								}
							}
						}
						else // 熟人
						{ 
							whatPeo = 0;
							if (conf < ans[i].conf) // 小于相似度，不相似（0.2~conf如0.45直接的人，不达标不要）
							{
								std::time_t nowtime = std::time(0);
								if (nowtime - ntime > 1)
								{
									whatPeo = 2; // 2 = 不达标人
									ans[i].fileName = "";
									it->second = t;
								}
							}

							else
							{
								ntime = t;
								if (t - it->second > (int)cam_config["config"]["target_continued_time"] && alg_name == "personnel_duty_detection") // 熟人
								{
									alarm = 1;
									it->second = (time_t)(t + (oldtime * 60.0));
									std::cout << "\033[34m" << "\e[1m" << "[Info] alarm:" << alg_name
											<< ",nowTime:" << st.tm_year + 1900 << "-" << st.tm_mon + 1 << "-" << st.tm_mday << " " << st.tm_hour << ":" << st.tm_min << ":" << st.tm_sec
											<< "," << rtsp_path << "\e[0m" << std::endl;
								}
							}
						}
						std::time_t nowtime = std::time(0);
						if (nowtime - ntime_1 > 2){
							time_interval = t;
						}
					}
					json jboxone2;
					json jboxone;
					int jx = (float)ans[i].x * ((float)1280.f / (float)frame.cols);
					int jw = (float)ans[i].w * ((float)1280.f / (float)frame.cols);
					int jy = (float)ans[i].y * ((float)(float)previewH / frame.rows);
					int jh = (float)ans[i].h * ((float)(float)previewH / (float)frame.rows);
					jboxone.emplace_back(jx);
					jboxone.emplace_back(jy);
					jboxone.emplace_back(jx + jw);
					jboxone.emplace_back(jy + jh);

					jboxone2["boxes"] = jboxone;
					jboxone2["whatPeo"] = whatPeo;
					jboxone2["score"] = ans[i].score;
					if (alg_name == "face_tracking")
						jboxone2["feature"] = ans[i].feature;
					// jboxone2["conf"] = ans[i].conf;//距离
					jboxone2["conf"] = int((1 - ans[i].conf / 2) * 100) / 100.0; // 相似度
					jboxone2["fileName"] = ans[i].fileName;
				  
					if (alarm == 1) // 报警
					{
						
						if (cam_config["roi"].size() == 2){
							//防盗检测相关
							int terminal = 0;
							std::vector<cv::Point> contours;
							json jROI = cam_config["roi"][1];
							for (int i = 0; i < jROI.size(); i++){
								contours.push_back(cv::Point(jROI[i][0], jROI[i][1]));
							}
							
							cv::Point CenterPoint = cv::Point((jx + jw / 2), (jy + jh / 2));
							if (cv::pointPolygonTest(contours, CenterPoint, true) > 0) 
							{
								terminal = 1;
							}
							jboxone2["touch_terminal"] = terminal;
						}
						
						if (whatPeo == 1){										 // 陌生人
							jboxone2["dur_time"] = (int)(t - time_start_dict1); // 陌生人持续时间;
							jboxone2["strangerId"]= -1;

						}
						else{
							jboxone2["dur_time"] = (int)(t - faceIdTimeMap[ans[i].id]); // 熟人持续时间
							jboxone2["spId"] = (int)ans[i].spId;
						}
						jwarn.emplace_back(jboxone2);
					}
					else
					{
						if (whatPeo == 1){
							jboxone2["dur_time"] = (int)(t - time_start_dict1); 		// 陌生人持续时间;
							jboxone2["strangerId"]= -1;
						}
						else if (whatPeo == 0){ 									  // 不报警，持续时间不满足
							jboxone2["dur_time"] = (int)(t - faceIdTimeMap[ans[i].id]); // 持续时间
							jboxone2["spId"] = (int)ans[i].spId;
						}
						else{															 //  人脸质量不达标
							jboxone2["dur_time"] = 0;
							jboxone2["spId"] = -1;
						}
						jresult.emplace_back(jboxone2);
					}
				}
			}
		}

		if (cam_config_open == 1 || (cam_config_open == 0 && alarm == 1))
		{
			makeSendMsg(frame, alarm, jresult, jwarn, cam_config["roi"]);
		}
		return;
	}
	
	void FaceProcess(outDetectData postmsg, json cam_config, int cam_config_open)
	{
		// ------------ 基本输入 ------------
		std::vector<faceInfo> ans = postmsg.faceinfos;
		cv::Mat frame = postmsg.img;
		int alarm = 0;
		json jresult;
		json jwarn;
	
		// 时间与打印
		std::time_t t = std::time(nullptr);
		struct tm st = {0};
		localtime_r(&t, &st);
	
		int previewH = 0;
		if (!frame.empty() && frame.cols > 0) {
			previewH = static_cast<int>(1280.f / static_cast<float>(frame.cols) * static_cast<float>(frame.rows));
			if (previewH % 2 != 0) previewH++;
		}
	
		// ------------ 配置读取（带默认值 + 类型安全）------------
		const auto& jcfg = cam_config.contains("config") ? cam_config["config"] : json::object();
		const float scoreThr = jcfg.value("score", 0.40f);
		const int	limArea  = jcfg.value("area", 0);
		// 将相似度阈值 conf(0~1, 越大要求越严格) 转成距离域 D(0~2, 越小越相似)
		float confD = (1.f - jcfg.value("conf", 0.45f)) * 2.f;
		if (confD < 0.f) confD = 0.f; if (confD > 2.f) confD = 2.f;
	
		const int  target_continued_time = std::max(0, jcfg.value("target_continued_time", 1)); // 秒
		const int  cooldown_minutes 	 = std::max(0, jcfg.value("time", 5));					// 报警后冷却分钟
		const int  strangerD_cut		 = 1; // 占位无用，仅示意
		(void)strangerD_cut;
	
		// ------------ 工作时间外，直接发送普通结果并返回 ------------
		if (!statusInTimes(jcfg.value("work_time", json::array()))) {

			return;
		}
	
		// ------------ 配置变更导致的重置（特别是 cooldown_minutes）------------
		if (oldtime < 0) oldtime = cooldown_minutes;
		if (std::abs(oldtime - (float)cooldown_minutes) > 1e-7f) {
			std::cout << "--[INFO] time is reset: " << alg_name << " " << rtsp_path << std::endl;
			oldtime = cooldown_minutes;
			// 陌生人持续计时器/冷却窗口重置到当前
			time_start_dict1 = t; 
			// 熟人计时清空
			faceIdTimeMap.clear();
		}
	
		// 初始化陌生人计时器：使用“<= t”为“未在冷却”的明确标志
		if (time_start_dict1 == 0) time_start_dict1 = t;
	
		// ------------ 清理过期的熟人计时条目 ------------
		// 说明：我们将 faceIdTimeMap[id] 的语义定义为：
		//	 若 <= 当前时间 t：这是该 id 的“开始计时时间点”
		//	 若  > 当前时间 t：这是该 id 的“冷却截止时间点”（冷却中）
		for (auto it = faceIdTimeMap.begin(); it != faceIdTimeMap.end(); ) {
			time_t stamp = it->second;
			if (stamp <= t) {
				// 在计时阶段，若“开始计时点”已远超阈值 + 3s（且还没触发过报警），视为过期清理
				if ((t - stamp) > (target_continued_time + 3)) {
					it = faceIdTimeMap.erase(it);
					continue;
				}
			}
			// 冷却阶段 (stamp > t) 不清理
			++it;
		}
	
		// ------------ 主循环：判定陌生人/熟人、计时/触发/冷却 ------------
		bool stranger_seen_this_frame = false;
	
		for (int i = 0; i < (int)ans.size(); ++i) {
//			// 先做基本质量/面积/score 过滤
//			if (ans[i].score < scoreThr) {
//				// 质量不达标，直接作为 whatPeo = 2
//			}
	
			int whatPeo = 0; // 0=熟人，1=陌生人，2=不达标
			// 距离域 D（0~2）：越小越相似；confD 是阈值，D <= confD 认为“像数据库里的人”
			const bool is_similar_to_db = (ans[i].conf <= confD);
			const bool in_db = (ans[i].id != -2);  // -2 表示数据库中没有该人
			const bool is_deleted = (ans[i].id == -1); // -1 表示需要删除的人
	
			// ========== 判定陌生人 / 熟人 ==========
			if (!in_db || !is_similar_to_db) {
				// 陌生人分支
				whatPeo = 1;
				ans[i].fileName.clear();
				stranger_seen_this_frame = true;
				ntime = t;
				// 陌生人冷却判断：若 time_start_dict1 > t，表示还在冷却期，直接跳过计时/报警
				if (time_start_dict1 <= t) {
					// 未在冷却：以 time_start_dict1 作为“连续出现起点”
					// 如果这是第一次看到陌生人且此前不是在“计时中”（即 time_start_dict1==0），上面已保障 !=0
					if ((t - time_start_dict1) > target_continued_time) {
						// 触发报警
						if (alg_name != "personnel_duty_detection"){
							alarm = 1;
							std::cout << "\033[34m" << "\e[1m" << "[Info] alarm:" << alg_name
									  << ",nowTime:" << (st.tm_year + 1900) << "-" << (st.tm_mon + 1) << "-" << st.tm_mday
									  << " " << st.tm_hour << ":" << st.tm_min << ":" << st.tm_sec
									  << "," << rtsp_path << "\e[0m" << std::endl;
						}
						// 进入冷却：下次允许报警的时间点
						time_start_dict1 = (time_t)(t + (oldtime * 60.0));
					}
				}
	
				// whatPeo==1 时，其它字段下方统一填充
			} else {
				// 熟人分支
				if (is_deleted) {
					// 标记删除的人视为不达标
					whatPeo = 2;
					ans[i].fileName.clear();
				} else {
					// 熟人：根据相似度再做一次“达标/不达标”判定
					if (!is_similar_to_db && 1.6 > ans[i].conf) {
						// 0.2~conf 的区间全部当“不达标”			  //2s内没检查到符合要求的熟人则重置间隔时间
						std::time_t nowtime = std::time(0);
						if (nowtime - time_start_dict[std::to_string(ans[i].id)] > 2){
							whatPeo = 2;
							ans[i].fileName.clear();
							faceIdTimeMap[ans[i].id] = nowtime;
						}
					} else {
						
						whatPeo = 0; // 达标的熟人
						auto it = faceIdTimeMap.find(ans[i].id);
						time_start_dict[std::to_string(ans[i].id)] = t;
						if (it == faceIdTimeMap.end()) {
							// 第一次看到该熟人 -> 开始计时（从当前 t 起）
							faceIdTimeMap[ans[i].id] = t;
						} else {
							time_t stamp = it->second;
							if (stamp > t) {
								// 冷却中（刚刚对该熟人触发过报警），不计时、也不触发
//								LOG_DEBUG("冷却中 stamp > t");
							} else {
								// 计时中
								if ((t - stamp) > target_continued_time && (alg_name == "personnel_duty_detection" || alg_name == "face_detection") ) {
									// 触发报警
									alarm = 1;
									std::cout << "\033[34m" << "\e[1m" << "[Info] alarm:" << alg_name
											  << ",nowTime:" << (st.tm_year + 1900) << "-" << (st.tm_mon + 1) << "-" << st.tm_mday
											  << " " << st.tm_hour << ":" << st.tm_min << ":" << st.tm_sec
											  << "," << rtsp_path << "\e[0m" << std::endl;
									// 进入冷却
									it->second = (time_t)(t + (oldtime * 60.0));
//									LOG_INFO("person conf = " + std::to_string(it->first) + ", " + std::to_string(it->second));
//									LOG_DEBUG("current time = " + std::to_string(t) + "/ new person time = " + std::to_string(it->second) + "/ old person time" + std::to_string(stamp));
								}
							}
						}
					}
				}
			}
	
			// ------------ 结果打包（框、类型、时间、相似度显示等）------------
			json jboxone2;
			json jbox;
			int jx = (float)ans[i].x * (1280.f / (float)std::max(1, frame.cols));
			int jw = (float)ans[i].w * (1280.f / (float)std::max(1, frame.cols));
			int jy = (float)ans[i].y * ((float)std::max(2, previewH) / (float)std::max(1, frame.rows));
			int jh = (float)ans[i].h * ((float)std::max(2, previewH) / (float)std::max(1, frame.rows));
			jbox.emplace_back(jx); jbox.emplace_back(jy); jbox.emplace_back(jx + jw); jbox.emplace_back(jy + jh);
	
			jboxone2["boxes"]	 = jbox;
			jboxone2["whatPeo"]  = whatPeo;
			jboxone2["score"]	 = ans[i].score;
			if (alg_name == "face_tracking")
				jboxone2["feature"] = ans[i].feature;
			jboxone2["conf"]	 = int((1 - ans[i].conf / 2) * 100) / 100.0; // 显示相似度(0~1)
			jboxone2["fileName"] = ans[i].fileName;
	
			// 触摸终端机（可选 ROI 第二组）
			if (cam_config.contains("roi") && cam_config["roi"].is_array() && cam_config["roi"].size() == 2) {
				int terminal = 0;
				std::vector<cv::Point> contours;
				json jROI = cam_config["roi"][1];
				for (int k = 0; k < (int)jROI.size(); ++k) {
					contours.emplace_back(jROI[k][0], jROI[k][1]);
				}
				cv::Point centerPt(jx + jw / 2, jy + jh / 2);
				if (!contours.empty() && cv::pointPolygonTest(contours, centerPt, true) > 0) terminal = 1;
				jboxone2["touch_terminal"] = terminal;
			}
	
			// 持续时间/ID 打包
			if (whatPeo == 1) {
				// 陌生人
				int dur = 0;
				if (time_start_dict1 <= t) 
					dur = (int)(t - time_start_dict1); // 非冷却期才显示持续时长
					
				jboxone2["dur_time"]   = dur;
				jboxone2["strangerId"] = -1;
			} else if (whatPeo == 0) {
				// 熟人
				int dur = 0;
				auto it = faceIdTimeMap.find(ans[i].id);
				if (it != faceIdTimeMap.end() && it->second <= t) 
					dur = (int)(t - it->second);
				
				jboxone2["dur_time"] = dur;
				jboxone2["spId"]	 = (int)ans[i].spId;
			} else {
				// 不达标
				jboxone2["dur_time"] = 0;
				jboxone2["spId"]	 = -1;
			}
	
			// 根据是否报警分别放入 jwarn / jresult
			if (alarm == 1) {
				jwarn.emplace_back(jboxone2);
			} else {
				jresult.emplace_back(jboxone2);
			}
		} // end for ans
	
		// 若本帧没有看到陌生人，且当前不处于冷却（time_start_dict1 <= t），且持续2s没检测到陌生人，则把陌生人计时器清零，避免“误续接”
		std::time_t nowtime = std::time(0);
		if (!stranger_seen_this_frame && time_start_dict1 <= t && (nowtime - ntime > 2) ) {
			time_start_dict1 = 0; // 下次再见到陌生人会重新从 0 开始计时
		}
	
		// ------------ 发送 ------------
		if (cam_config_open == 1 || (cam_config_open == 0 && alarm == 1)) {
			makeSendMsg(frame, alarm, jresult, jwarn, cam_config["roi"]);
		}
		return;
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
    time_t ntime_1 = 0;
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

    // 人脸
    float stranger_alarm_time = -1;
    std::map<idx_t, time_t> faceIdTimeMap;

    // ws
    net::io_context ioc;
    tcp::resolver resolver{ioc};
    websocket::stream<tcp::socket> ws{ioc};
};
