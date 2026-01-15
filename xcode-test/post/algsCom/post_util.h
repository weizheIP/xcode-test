#pragma once

#include "Track/inc/SORT.h"
#include "base.h"
#include "httpprocess.h"

#include "base64.h"
#include "json.h"
#include <iostream>
#include "string.h"
#include <mutex>
#include <unistd.h>
#include <sys/time.h>
#include <stdio.h>
#include <vector>
#include "string.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <cmath>
// using namespace std;

#include "common/json/json.hpp"
using json = nlohmann::json;

static std::string sanitizeRtspUrl(const std::string& url) {
    std::string sanitized = url;
    
    // 替换所有非法字符为下划线
    const std::string illegalChars = ":/?=&%.";
    for (char& c : sanitized) {
        if (illegalChars.find(c) != std::string::npos) {
            c = '_';
        }
    }

    return sanitized;
}

static std::map<std::string, int> mCOLORList{
    {"blue", 0},
    {"other", 1},
    {"black", 2},
    {"green", 3},
    {"white", 4},
    {"red", 5},
    {"yellow", 6},
    {"pink", 7},
    {"orange", 8},
    {"brown", 9},
    {"purple", 10}};

#include <chrono>
#include <iomanip>
#include <sstream>

static std::string getCurrentTimeFormatted()
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

static void update_camera_status(std::string rtsp_path, int status, std::string errMsg)
{
    std::string data, sendStr;
    Json::Value ans;
    ans["rtsp"] = rtsp_path;
    ans["status"] = status;
    ans["errMsg"] = errMsg;
    sendStr = ans.toStyledString();
    HttpProcess http("http://127.0.0.1:9696");
    http.Post_Json("/api/camera/status", sendStr, data);
    // std::cout << "update camera status: "<<rtsp_path << "--"<< status << "--" << data << std::endl;
}

static void update_camera_status(std::string rtsp_path, int status, std::string errMsg, std::string img)
{
    std::string data, sendStr;
    Json::Value ans;
    ans["rtsp"] = rtsp_path;
    ans["status"] = status;
    ans["errMsg"] = errMsg;
    ans["img"] = img;
    sendStr = ans.toStyledString();
    HttpProcess http("http://127.0.0.1:9696");
    http.Post_Json("/api/camera/status", sendStr, data);
    // std::cout << "update camera status: "<<rtsp_path << "--"<< status << "--" << data << std::endl;
}

static int mode_list(std::vector<int> v1)
{
    if (v1.begin() == v1.end())
        return 0;
    sort(v1.rbegin(), v1.rend());
    int i, t, x;
    std::vector<int>::iterator iter1 = v1.begin();
    i = count(v1.begin(), v1.end(), *iter1);
    t = *iter1;
    for (iter1++; iter1 != v1.end(); iter1++)
    {
        x = count(v1.begin(), v1.end(), *iter1);
        if (x > i)
        {
            i = x;
            t = *iter1;
        }
    }
    return t;
}

static bool checkGreenScreen(const uint8_t *buffer, int width, int height)
{
    if (buffer == NULL || width < 0 || height < 0)
    {
        return false;
    }
    int x1 = rand() % (width - 9);
    int y1 = rand() % (height / 2 - 9);

    int x2 = rand() % (width - 9);
    int y2 = rand() % (height / 2 - 9) + height / 2;

    int zeroCount = 0;
    for (int j = y1; j < y1 + 10; j++)
    {
        for (int i = x1; i < x1 + 10; i++)
        {
            if (buffer[j * width + i] == 0)
                zeroCount++;
        }
    }

    for (int j = y2; j < y2 + 10; j++)
    {
        for (int i = x2; i < x2 + 10; i++)
        {
            if (buffer[j * width + i] == 0)
                zeroCount++;
        }
    }

    if (zeroCount > 150)
        return true;
    return false;
}

static void cv_encode_base64(const cv::Mat& img, string &img64)
{
    if(img.empty())
        return;
    std::vector<unsigned char> buf;
    cv::imencode(".jpg", img, buf);
    img64 = base64_encode(&(buf[0]), buf.size());
    return;
}

static float GET_IOU2(int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2)
{
    float lx = x1 > x2 ? x1 : x2;
    float ly = y1 > y2 ? y1 : y2;
    float rx = (x1 + w1) < (x2 + w2) ? (x1 + w1) : (x2 + w2);
    float ry = (y1 + h1) < (y2 + h2) ? (y1 + h1) : (y2 + h2);

    // if(lx > rx || rx > ry) return 0;

    if (lx > rx || ly > ry)
        return false;
    return true;
}

static bool isInRoi(BoxInfo car,std::vector<cv::Point> roiContour)
{
    int xc = (car.x1+car.x2)/2;
    int yc09 = car.y2 - ((car.y2-car.y1)/10);
    int yc08 = car.y2 - (2 * (car.y2-car.y1)/10);
    cv::Point centPointOfBottom(xc,yc09);
    cv::Point centPointOfBottom08(xc,yc08);

    bool isIn = (cv::pointPolygonTest(roiContour, centPointOfBottom, true) >= 0) || (cv::pointPolygonTest(roiContour, centPointOfBottom08, true) >= 0);
    return isIn;
}

void isTouchTerminal(int &terminal, std::vector<cv::Point> contours, int jx, int jy, int jw, int jh)
{

	cv::Point CenterPoint = cv::Point((jx + jw / 2), (jy + jh / 2));
	if (cv::pointPolygonTest(contours, CenterPoint, true) > 0) 
	{
		terminal = 1;
	}
}

static void RoiFillter(vector<BoxInfo> &detectResults, vector<vector<int>> jROI, std::string algsName)
{
    if (jROI.size() == 0)
        return;

    std::vector<cv::Point> contours;

    for (int i = 0; i < jROI.size(); i++)
        contours.push_back(cv::Point(jROI[i][0], jROI[i][1])); // 把json数据的第 0 1 数据组装成 cv::point 放到 contours

    int t = detectResults.size();

    for (int i = 0; i < t; i++) // 遍历原有的框框图，把符合roi的标准筛选出来
    {
        for (std::vector<BoxInfo>::const_iterator r_ = detectResults.begin(); r_ != detectResults.end(); r_++)
        {
            bool isInRoi = false;
            if (algsName == "intrusion_detection" || 
                algsName == "over_the_wall_detection" ||
                algsName == "cv_perimeter_invasion" || 
                algsName == "climbing_recognition") // 
            {
                int xc = (r_->x1 + r_->x2) / 2;
                int yc09 = r_->y2;
                cv::Point centPointOfBottom(xc, yc09);
                isInRoi = (cv::pointPolygonTest(contours, centPointOfBottom, true) < 0);
            }
            else
            {
                cv::Point CenterPoint((r_->x1 + r_->x2) / 2, (r_->y1 + r_->y2) / 2);
                isInRoi = (cv::pointPolygonTest(contours, CenterPoint, true) < 0);
            }

            if (isInRoi) // 用于测试一个点是否在多边形中，contours是上面输入roi
            {
                detectResults.erase(r_);
                break;
            }
        }
    }
}

static bool is_3d_array(const json& j) {
    if (!j.is_array()) return false;
    for (const auto& elem : j) {
        if (!elem.is_array()) return false;
        for (const auto& sub_elem : elem) {
            if (!sub_elem.is_array()) return false;
            // for (const auto& val : sub_elem) {
            //     if (!val.is_number_integer()) return false;
            // }
        }
    }
    return true;
}

static void MultiRoiFillter(vector<BoxInfo> &detectResults, vector<vector<vector<int>>> jROI, std::string algsName)
{
    if (jROI.size() == 0)
        return;

    vector<vector<cv::Point>> contours;
    for (int i = 0; i < jROI.size(); i++)
    {
        std::vector<cv::Point> contour;
        for (int j = 0; j < jROI[i].size(); j++)
            contour.push_back(cv::Point(jROI[i][j][0], jROI[i][j][1]));
        contours.push_back(contour);
    }
  
    int t = detectResults.size();

    for (int i = 0; i < t; i++) // 遍历原有的框框图，把符合roi的标准筛选出来
    {
        for (std::vector<BoxInfo>::const_iterator r_ = detectResults.begin(); r_ != detectResults.end(); r_++)
        {
            bool isInRoi = false;
            
            for (const auto& contour : contours) {
                cv::Point CenterPoint((r_->x1 + r_->x2) / 2, (r_->y1 + r_->y2) / 2);
                double distance = cv::pointPolygonTest(contour, CenterPoint, false);
                if (distance >= 0) {  // 点在轮廓内部或边缘
                    isInRoi= true;
                }
            }               
            if (!isInRoi) // 用于测试一个点是否在多边形中，contours是上面输入roi
            {
                detectResults.erase(r_);
                break;
            }
        }
    }
}

//static bool StatusIntTime(std::string beginT, std::string endT)
//{
//    time_t rawtime;
//    struct tm *info;
//    time(&rawtime);
//
//    info = localtime(&rawtime);
//    int shi = info->tm_hour;
//    int fen = info->tm_min;
//    // cout<<shi<<":"<<fen<<endl;
//
//    int timeGetter[4] = {0};
//    int index = beginT.find(":");
//
//    timeGetter[0] = atoi(beginT.substr(0, index).c_str());
//    timeGetter[1] = atoi(beginT.substr(index + 1, beginT.size()).c_str());
//
//    timeGetter[2] = atoi(endT.substr(0, index).c_str());
//    timeGetter[3] = atoi(endT.substr(index + 1, endT.size()).c_str());
//
//    int Time_min[3] = {0};
//    Time_min[0] = timeGetter[0] * 60 + timeGetter[1];
//    Time_min[1] = timeGetter[2] * 60 + timeGetter[3];
//    Time_min[2] = info->tm_hour * 60 + info->tm_min;
//
//    if (Time_min[0] > Time_min[1]) // 23:00 - 5:00 跨天情况
//    {
//        if (Time_min[2] > Time_min[0] || Time_min[2] < Time_min[1])
//            return true;
//    }
//    else
//    {
//        if (Time_min[2] > Time_min[0] && Time_min[2] < Time_min[1])
//        {
//            // if(info->tm_min > timeGetter[1] && info->tm_min < timeGetter[3])
//            return true;
//        }
//    }
//
//    return false;
//}

// 将HH:MM:SS格式的时间转换为总秒数
int time_to_seconds(const std::string& time_str) {
    int hours, minutes, seconds;
    char colon;
    std::istringstream iss(time_str);
    
    iss >> hours >> colon >> minutes >> colon >> seconds;
    
    if (iss.fail() || colon != ':') {
        return -1; // 格式错误
    }
    
    return hours * 3600 + minutes * 60 + seconds;
}

static bool StatusIntTime(std::string beginT, std::string endT)
{
    // 获取当前时间戳
    std::time_t t = std::time(nullptr);
    std::tm* now = std::localtime(&t);

    // 当前时分秒
    int hours   = now->tm_hour;
    int minutes = now->tm_min;
    int seconds = now->tm_sec;

    // 转换为总秒数
    int current_seconds = hours * 3600 + minutes * 60 + seconds;

	int start_seconds = time_to_seconds(beginT);
	int end_seconds = time_to_seconds(endT);
	
	if (start_seconds == -1 || end_seconds == -1) return false; // 跳过无效区间
	
	// 处理跨天的情况（结束时间小于开始时间表示跨天）
	if (end_seconds < start_seconds) {
		// 跨天区间：当前时间 >= 开始时间 OR 当前时间 <= 结束时间
		if (current_seconds >= start_seconds || current_seconds <= end_seconds) {
			return true;
		}
	} else {
		// 正常区间：开始时间 <= 当前时间 <= 结束时间
		if (current_seconds >= start_seconds && current_seconds <= end_seconds) {
			return true;
		}
	}
    return false;
}


// static bool statusInTimes(vector<vector<string>> timeArr)
static bool statusInTimes(json timeArr1)
{
   
    // if (timeArr.size() <= 0)
    //     return false;
    // if(timeArr.size()==1)
    if(!timeArr1[0].is_array())
    {
        vector<string> timeArr=timeArr1;
        return StatusIntTime(timeArr[0],timeArr[1]);
    }
    else
    {
        vector<vector<string>> timeArr=timeArr1;
        bool isNotTime = false;
        for (int i = 0; i < timeArr.size(); i++)
        {
            if (StatusIntTime(timeArr[i][0], timeArr[i][1]))
            {
                isNotTime = true;
                break;
            }
        }
        return isNotTime;
    }
    return false;
}

static bool no_ID_fire_smoking_detect(BoxInfo &oldBox, BoxInfo boxNew, int limArea)
{
    if (oldBox.x1 == 0 && oldBox.x2 == 0)
    {
        oldBox.x1 = boxNew.x1;
        oldBox.x2 = boxNew.x2;
        oldBox.y1 = boxNew.y1;
        oldBox.y2 = boxNew.y2;
        return false;
    }
    float height1 = oldBox.y2 - oldBox.y1;
    float height2 = boxNew.y2 - boxNew.y1;
    float area = (float)std::abs(boxNew.y2 - boxNew.y1) * std::abs(boxNew.x2 - boxNew.x1);
    float drift = (float)std::abs(height1 - height2) / height2;

    if (drift >= 0.05 && area >= (limArea * limArea))
    {
        oldBox.x1 = boxNew.x1;
        oldBox.x2 = boxNew.x2;
        oldBox.y1 = boxNew.y1;
        oldBox.y2 = boxNew.y2;
        return true;
    }
    return false;
}

static bool same_place_detect(BoxInfo &oldBox, BoxInfo boxNew)
{
    if (oldBox.x1 == 0 && oldBox.x2 == 0)
    {
        oldBox.x1 = boxNew.x1;
        oldBox.x2 = boxNew.x2;
        oldBox.y1 = boxNew.y1;
        oldBox.y2 = boxNew.y2;
        return false;
    }
    bool isDiff = (std::abs(boxNew.x1 - oldBox.x1) > 100 || std::abs(boxNew.x2 - oldBox.x2) > 100 || std::abs(boxNew.y1 - oldBox.y1) > 100 || std::abs(boxNew.y2 - oldBox.y2) > 100);
    if (isDiff)
        return true;
    else
        return false;
}

static bool same_place_detect_arr(std::vector<BoxInfo> &oldBoxs, BoxInfo &boxNew)
{
    if (oldBoxs.size() == 0)
        return false;
    bool isSamePlace = false;
    for (auto x : oldBoxs)
    {
        int isDiff = (std::abs(boxNew.x1 - x.x1) + std::abs(boxNew.x2 - x.x2) + std::abs(boxNew.y1 - x.y1) + std::abs(boxNew.y2 - x.y2));
        // printf("[warn] %d\n",isDiff);
        if (isDiff < 150)
        {
            // printf("[warn] isSamePlace!\n");
            isSamePlace = true;
            break;
        }
    }
    return isSamePlace;
}

static bool same_place_detect_arr2(std::vector<SORT::TrackingBox> &oldBoxs, SORT::TrackingBox &boxNew)
{
    if (oldBoxs.size() == 0)
        return false;
    bool isSamePlace = false;
    for (auto x : oldBoxs)
    {
        int isDiff = (std::abs(boxNew.box.x - x.box.x) + std::abs(boxNew.box.y - x.box.y) + std::abs(boxNew.box.width - x.box.width) + std::abs(boxNew.box.height - x.box.height));
        // printf("[warn] %d\n",isDiff);
        if (isDiff < 150)
        {
            // printf("[warn] isSamePlace!\n");
            isSamePlace = true;
            break;
        }
    }
    return isSamePlace;
}

static void getGiggerOutSizeReat2(int &x, int &y, int &w, int &h, vector<vector<int>> roi1, vector<vector<int>> roi2)
{
    int maxX = -1, minX = 9999, maxY = -1, minY = 9999;
    for (int i = 0; i < roi1.size(); i++)
    {
        int tempx = roi1[i][0];
        if (tempx > maxX)
            maxX = tempx;
        if (tempx < minX)
            minX = tempx;
        tempx = roi1[i][1];
        if (tempx > maxY)
            maxY = tempx;
        if (tempx < minY)
            minY = tempx;
    }
    for (int i = 0; i < roi2.size(); i++)
    {
        int tempx = roi2[i][0];
        if (tempx > maxX)
            maxX = tempx;
        if (tempx < minX)
            minX = tempx;
        tempx = roi2[i][1];
        if (tempx > maxY)
            maxY = tempx;
        if (tempx < minY)
            minY = tempx;
    }
    x = minX;
    y = minY;
    w = maxX - minX;
    h = maxY - minY;
}

// static void getGiggerOutSizeReat2(int &x, int &y, int &w, int &h, Json::Value roi1, Json::Value roi2)
// {
//     int maxX = -1, minX = 9999, maxY = -1, minY = 9999;
//     for (int i = 0; i < roi1.size(); i++)
//     {
//         int tempx = roi1[i][0].asInt();
//         if (tempx > maxX)
//             maxX = tempx;
//         if (tempx < minX)
//             minX = tempx;
//         tempx = roi1[i][1].asInt();
//         if (tempx > maxY)
//             maxY = tempx;
//         if (tempx < minY)
//             minY = tempx;
//     }
//     for (int i = 0; i < roi2.size(); i++)
//     {
//         int tempx = roi2[i][0].asInt();
//         if (tempx > maxX)
//             maxX = tempx;
//         if (tempx < minX)
//             minX = tempx;
//         tempx = roi2[i][1].asInt();
//         if (tempx > maxY)
//             maxY = tempx;
//         if (tempx < minY)
//             minY = tempx;
//     }
//     x = minX;
//     y = minY;
//     w = maxX - minX;
//     h = maxY - minY;
// }

static bool leave_post_detect_(vector<BoxInfo> &generate_boxes, int limCnt)
{
    int bodyCnt = 0;
    int headCnt = 0;
    for (int i = 0; i < generate_boxes.size(); i++)
    {
        if (generate_boxes[i].label == 0)
            bodyCnt++;
        else
            headCnt++;
    }
    return !((bodyCnt >= limCnt) || (headCnt >= limCnt));
}

static bool smocking_detect2(vector<BoxInfo> box0, vector<BoxInfo> box1) // 0吸烟1人2头
{
    for (auto item : box0)
    {
        if (item.label == 0)
        {
            for (auto item2 : box1)
            {
                if (item2.label == 2 && item2.score > 0.4)
                {
                    int h1 = item.y2 - item.y1;
                    int w1 = item.x2 - item.x1;

                    int h2 = item2.y2 - item2.y1;
                    int w2 = item2.x2 - item2.x1;
                    float areaRatio = abs(float(h1 * w1) / float(h2 * w2)); // 面积大小
                    if (areaRatio > 0.8)
                        return false;
                    // if(0.001 < GET_IOU2(item.x1,item.y1,w1,h1,item2.x1,item2.y1,w2,h2)) // 交集
                    if (true == GET_IOU2(item.x1, item.y1, w1, h1, item2.x1, item2.y1, w2, h2)) // 交集
                    {
                        // std::cout <<"smoking: "<< item.x1<<"--"<<item.y1<<"--"<<item.x2<<"--"<<item.y2<<"--head: "<<item2.x1<<"--"<<item2.y1<<"--"<<item2.x2<<"--"<<item2.y2 << std::endl;
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

static int garbage_detect2(vector<BoxInfo> box1, vector<BoxInfo> box2, vector<BoxInfo> &box3)
{ // 0 无目标 1 报警 2有目标但有交集
    if (box1.size() <= 0)
        return 0;

    bool isIoU = true;
    for (auto item : box1)
    {
        bool itemIou = true;
        for (auto item2 : box2)
        {
            if (item2.label == 4 && item2.score > 0.4)
            { // 0123 垃圾 4人5头
                int h1 = item.y2 - item.y1;
                int w1 = item.x2 - item.x1;

                int h2 = item2.y2 - item2.y1;
                int w2 = item2.x2 - item2.x1;
                // float areaRatio = abs(float(h1*w1)/float(h2*w2)); //面积大小
                // if(areaRatio > 0.5)
                //     return false;
                if (0.01 < GET_IOU2(item.x1, item.y1, w1, h1, item2.x1, item2.y1, w2, h2)) // 有交集
                {
                    // std::cout<<"iou :"<<GET_IOU2(item.x1,item.y1,w1,h1,item2.x1,item2.y1,w2,h2)<<"\n";
                    itemIou = false;
                    break;
                }
            }
        }
        if (itemIou)
        {
            box3.push_back(item);
            isIoU = false;
        }
    }
    if (!isIoU)
        return 1;
    else
        return 2;
}
// 假定已有 struct BoxInfo { int x1,y1,x2,y2,label; /* 0手机 1手 */ };
// 假定已有 GET_IOU2(...): bool  // 返回是否相交/重叠（你原来的函数）

static inline bool pointInPolygonFast(const std::vector<cv::Point>& poly,
                                      const cv::Rect& bbox,
                                      const cv::Point& p)
{
    if (!bbox.contains(p)) return false;                 // 先用外接矩形快速剔除
    return cv::pointPolygonTest(poly, p, /*measureDist*/true) > 0;
}

static inline int64_t sqrll(int64_t x) { return x * x; }

static bool illegal_operation_phone(std::vector<BoxInfo> box0,
                                    std::vector<std::vector<int>> jROI,
                                    int dist_mul) // 0手机 1手
{
    if (jROI.empty()) return false;

    // 1) 组装 ROI 多边形与外接矩形（后者用于快速过滤）
    std::vector<cv::Point> contours;
    contours.reserve(jROI.size());
    for (auto& p : jROI) contours.emplace_back(p[0], p[1]);
    const cv::Rect roiBBox = cv::boundingRect(contours);

    // 2) 拆分手机、收集手；同时预计算“手”的中心点与宽度
    std::vector<BoxInfo> phones;
    phones.reserve(box0.size());

    struct HandFeat {
        BoxInfo b;
        int cx, cy;      // 中心点
        int width;       // 手框宽度
        bool hasPhone;   // 与手机有交集
        bool inROI_noPhone; // 在ROI且无手机交集
    };
    std::vector<HandFeat> hands; hands.reserve(box0.size());

    for (const auto& it : box0) {
        if (it.label == 0) {
            phones.push_back(it);
        } else if (it.label == 1) {
            HandFeat hf;
            hf.b = it;
            hf.cx = (it.x1 + it.x2) >> 1;
            hf.cy = (it.y1 + it.y2) >> 1;
            hf.width = std::max(1, it.x2 - it.x1);
            hf.hasPhone = false;
            hf.inROI_noPhone = false;
            hands.push_back(hf);
        }
    }
    if (hands.empty()) return false;

    // 3) 标记所有“与手机有交集的手”  (一次 H*P，命中即 break)
    for (auto& h : hands) {
        for (const auto& p : phones) {
            // 可选：先做 AABB 快过滤，减少 GET_IOU2 调用
            if (h.b.x2 <= p.x1 || h.b.x1 >= p.x2 || h.b.y2 <= p.y1 || h.b.y1 >= p.y2)
                continue;
            if (GET_IOU2(p.x1, p.y1, p.x2 - p.x1, p.y2 - p.y1,
                         h.b.x1, h.b.y1, h.b.x2 - h.b.x1, h.b.y2 - h.b.y1))
            {
                h.hasPhone = true;
//                break; // 命中即停止
            }
        }
    }

    // 4) 组成集合 A、B
    std::vector<int> setA; setA.reserve(hands.size()); // 手∩手机
    std::vector<int> setB; setB.reserve(hands.size()); // 手∩ROI 且 手∩手机为空
    for (int i = 0; i < (int)hands.size(); ++i) {
        auto& h = hands[i];
        if (h.hasPhone) {
            setA.push_back(i);
        } else {
            // 只对“可能在 ROI 内”的再做 pointPolygonTest
            if (pointInPolygonFast(contours, roiBBox, cv::Point(h.cx, h.cy))) {
                h.inROI_noPhone = true;
                setB.push_back(i);
            }
        }
    }
    if (setA.empty() || setB.empty()) return false;

    // 5) A×B 距离判定： dist^2 < (5 * handWidth)^2
    for (int ia : setA) {
        const auto& ha = hands[ia];
        const int64_t thr = 1LL * dist_mul * ha.width;        // “拿手机的手”的宽度
        const int64_t thr2 = thr * thr;            // 阈值平方
        for (int ib : setB) {
            const auto& hb = hands[ib];
            const int64_t dx = (int64_t)ha.cx - (int64_t)hb.cx;
            const int64_t dy = (int64_t)ha.cy - (int64_t)hb.cy;
            if (sqrll(dx) + sqrll(dy) < thr2) {
                return true; 
            }
        }
    }

    return false;
}

static int chooseBestCarPlateIndex(std::vector<CarBoxInfo>& vec,int boxHeight)
{
    int ans = -1;
    float maxScore = 0;
    for(int i = 0; i < vec.size(); i++)
    {
        if(vec[i].y1 < (boxHeight/3)) //车牌的位置还没变换，车牌的位置必须大于车框的高度的三分之一
            continue;
        if(vec[i].score > maxScore)
        {
            maxScore = vec[i].score;
            ans = i;
        }
    }

    return ans;
}

static cv::Mat get_split_merge(cv::Mat &img)   //双层车牌 分割 拼接
{
    cv::Rect  upper_rect_area = cv::Rect(0,0,img.cols,int(5.0/12*img.rows));
    cv::Rect  lower_rect_area = cv::Rect(0,int(1.0/3*img.rows),img.cols,img.rows-int(1.0/3*img.rows));
    cv::Mat img_upper = img(upper_rect_area);
    cv::Mat img_lower =img(lower_rect_area);
    cv::resize(img_upper,img_upper,img_lower.size());
    cv::Mat out(img_lower.rows,img_lower.cols+img_upper.cols, CV_8UC3, cv::Scalar(114, 114, 114));
    img_upper.copyTo(out(cv::Rect(0,0,img_upper.cols,img_upper.rows)));
    img_lower.copyTo(out(cv::Rect(img_upper.cols,0,img_lower.cols,img_lower.rows)));
    return out;
}

// static void RoiFillter_car(std::shared_ptr<InferOutputMsg> data, std::vector<cv::Point> contours)
static void RoiFillter_car(vector<BoxInfo>& data_result, std::vector<cv::Point> contours)
{
    // std::string strs = jROI.toStyledString();
    // std::cout<<strs<<std::endl;
    if (contours.size() == 0)
        return;

    // std::vector<cv::Point> contours;

    // for (int i = 0; i < jROI.size(); i++)
    //     contours.push_back(cv::Point(jROI[i][0].asInt(), jROI[i][1].asInt()));//把json数据的第 0 1 数据组装成 cv::point 放到 contours
    // std::vector<int>::const_iterator iter = b_list->begin();iter!=b_list->end();iter++
    int t = data_result.size();

    for (int i = 0; i < t; i++) //遍历原有的框框图，把符合roi的标准筛选出来
    {
        for(int j = 0; j < data_result.size(); j++)
        {
            cv::Point CenterPoint((data_result[j].x1 + data_result[j].x2) / 2, (data_result[j].y1 + data_result[j].y2) / 2);
            if (cv::pointPolygonTest(contours, CenterPoint, true) < 0) //用于测试一个点是否在多边形中，contours是上面输入roi
            {
                data_result.erase(data_result.begin()+j);
                break;
            }
        }

    }
}
