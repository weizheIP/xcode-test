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

static std::map<std::string,int> mCOLORList{
    {"blue",0},
    {"other",1},
    {"black",2},
    {"green",3},
    {"white",4},
    {"red",5},
    {"yellow",6},
    {"pink",7},
    {"orange",8},
    {"brown",9},
    {"purple",10}
};


#include <chrono>
#include <iomanip>
#include <sstream>

static std::string getCurrentTimeFormatted() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

static void update_camera_status(std::string rtsp_path,int status){
    // update camera status...
    std::string data,sendStr;            
    Json::Value ans;
    ans["rtsp"] = rtsp_path;
    ans["status"] = status;
    sendStr = ans.toStyledString();
    HttpProcess http("http://127.0.0.1:9696");
    http.Post_Json("/api/camera/status",sendStr,data);
    std::cout<<"update camera status: "<<status<<"--"<<data<<std::endl;
}


static double roundToTwoDecimalPlaces(double number) {
    double scaleFactor = pow(10, 2);  // 设置小数位数
    return round(number * scaleFactor) / scaleFactor;
}


static int mode_list(std::vector<int> v1)
{
    if(v1.begin() == v1.end())
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

static void cv_encode_base64(const cv::Mat& img, string &img64)
{
    if(img.empty())
        return;
    std::vector<unsigned char> buf;
    cv::imencode(".jpg", img, buf);
    img64 = base64_encode(&(buf[0]), buf.size());
    return;
}


static bool checkGreenScreen(const uint8_t* buffer, int width, int height)
{
    if(buffer == NULL || width<0 || height< 0)
    {
        return false;
    }
    int x1 = rand() % (width - 9);
    int y1 = rand()  % (height / 2 - 9);

    int x2 = rand() % (width - 9);
    int y2 = rand()  % (height / 2 - 9) + height / 2;

    int zeroCount = 0;
    for(int j = y1; j < y1 + 10; j++)
    {
        for(int i = x1; i < x1 + 10; i++)
        {
            if(buffer[j * width + i] == 0)
                zeroCount++;
        }
    }

    for(int j = y2; j < y2 + 10; j++)
    {
        for(int i = x2; i < x2 + 10; i++)
        {
            if(buffer[j * width + i] == 0)
                zeroCount++;
        }
    }

    if(zeroCount > 150)
        return true;
    return false;
}

static void cv_encode_base64_compression(const cv::Mat& img, std::string &img64)
{
    std::vector<unsigned char> buf;
    std::vector<int> compression_params;
    compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
    compression_params.push_back(30);  // 设置压缩质量，范围为 0-100
    cv::imencode(".jpg", img, buf, compression_params);
    img64 = base64_encode(&(buf[0]), buf.size());
}

// static void cv_encode_base64(const cv::Mat& img, std::string &img64)
// {
//     std::vector<unsigned char> buf;    
//     cv::imencode(".jpg", img, buf);
//     img64 = base64_encode(&(buf[0]), buf.size());
// }

// 废弃
static std::string encode_base64(cv::Mat frame)
{
    std::vector<unsigned char> img_data;
    cv::imencode(".jpg", frame, img_data);   
    unsigned char imgchardata[img_data.size()];
    int siz= img_data.size()*sizeof(img_data[0]);
    memcpy(imgchardata,&(img_data[0]),siz);
    std::string img64 = base64_encode(imgchardata , siz);
    // base64data = img64;
    memset(imgchardata,0,sizeof(imgchardata));
    img_data.clear();
    return img64;
}

static void get_now_time(int &date,int &hour,int &min ,int &sec){
    time_t rawtime;
    struct tm *info;
    char buffer[80];

    time( &rawtime );

    info = localtime( &rawtime );
    date = info->tm_mday;
    hour = info->tm_hour;
    min = info->tm_min;
    sec = info->tm_sec;
}

static void get_format_time(char* buffer){
    time_t rawtime;
    struct tm *info;
    
    time( &rawtime );
    
    info = localtime( &rawtime );

    timeval t1;
    gettimeofday(&t1, NULL);
    // printf("%.3f",(float)(t2.tv_sec - t1.tv_sec)+(float)(t2.tv_usec-t1.tv_usec)/1000000.0f);
    
    strftime(buffer, 80, "[%Y-%m-%d %H:%M:%S]", info);
    std::stringstream sss;
    sss<<buffer<<"."<<(int)((t1.tv_usec)/1000.0f);
    strcpy(buffer,sss.str().c_str());
}


static void get_format_time2(char* buffer,std::string typeName){
    time_t rawtime;
    struct tm *info;
    time( &rawtime );
    info = localtime( &rawtime );
    timeval t1;
    gettimeofday(&t1, NULL);
    // printf("%.3f",(float)(t2.tv_sec - t1.tv_sec)+(float)(t2.tv_usec-t1.tv_usec)/1000000.0f);
    std::stringstream sss;
    sss<<buffer<<"/civi/civiapp/resource/";
    
    // sss<<buffer<<"/civi/main_frame/runtime/resource/";
    strftime(buffer, 200, "%Y-%02m-%02d/", info);
    sss<<buffer<<typeName;
    strftime(buffer, 200, "_%02H:%02M:%02S", info);
    sss<<buffer<<":"<<(int)((t1.tv_usec)/1000.0f)<<"ms.mp4";
    strcpy(buffer,sss.str().c_str());
}

static void get_format_time3(char* buffer,std::string typeName, int id,bool inOrOut){
    time_t rawtime;
    struct tm *info;
    time( &rawtime );
    info = localtime( &rawtime );
    timeval t1;
    gettimeofday(&t1, NULL);
    // printf("%.3f",(float)(t2.tv_sec - t1.tv_sec)+(float)(t2.tv_usec-t1.tv_usec)/1000000.0f);
    std::stringstream sss;
    sss<<buffer<<"/civi/civiapp/resource/";
    
    // sss<<buffer<<"/civi/main_frame/runtime/resource/";
    strftime(buffer, 200, "%Y-%02m-%02d/", info);
    sss<<buffer<<typeName;
    strftime(buffer, 200, "_%02H:%02M:%02S", info);
    sss<<buffer<<":"<<(int)((t1.tv_usec)/1000.0f)<<"ms_id:"<<id<<"_"<<(inOrOut?"out":"in")<<".mp4";
    strcpy(buffer,sss.str().c_str());
}

// Json::Value Get_Rtsp_Json(Json::Value &msg_setting_,std::string rtsp_path){
    
//     Json::Value ans;
//     if(!msg_setting_.isObject())
//         return ans;

//     for(int i =0 ;i < msg_setting_["data"].size();i++){
//         if(msg_setting_["data"][i]["algorithm"].asString()== sTypeName)
//         {
//             for(int j = 0;j<msg_setting_["data"][i]["cams"].size();j++)
//             {
//                 if(msg_setting_["data"][i]["cams"][j]["rtsp"] == rtsp_path)
//                      return msg_setting_["data"][i]["cams"][j];
//             }
//         }
           
//     }
//     return msg_setting_["data"][0];
// }

static Json::Value Get_Rtsp_Json(Json::Value &msg_setting_,std::string rtsp_path){
    // Json::Reader read;
    // Json::Value value;
    // read.parse( msg_setting_.toStyledString(),value);
    for(int i =0 ;i<msg_setting_["data"].size();i++){
        if(msg_setting_["data"][i]["rtsp"]==rtsp_path)
            return msg_setting_["data"][i];
    }
    return msg_setting_["data"][0];
}

static Json::Value Get_Rtsp_Json(std::shared_ptr<Json::Value> &msg_setting_,std::string rtsp_path){
    Json::Reader read;
    Json::Value value;
    read.parse( msg_setting_->toStyledString(),value);
    for(int i =0 ;i<value["data"].size();i++){
        if(value["data"][i]["rtsp"]==rtsp_path)
            return value["data"][i];
    }
    return value["data"][0];
}

static Json::Value Get_Rtsp_Json1(Json::Value &msg_setting_,std::string rtsp_path,std::string  c_typeName){
    Json::Reader read;
    Json::Value value;
    read.parse( msg_setting_.toStyledString(),value);
    for(int i =0 ;i<value["data"].size();i++){
        if(value["data"][i]["algorithm"].asString()== c_typeName)
        {
            for(int j = 0;j<value["data"][i]["cams"].size();j++)
            {
                if(value["data"][i]["cams"][j]["rtsp"] == rtsp_path)
                     return value["data"][i]["cams"][j];
            }
        }
           
    }
    return value["data"][0];
}


static bool isIntersect(BoxInfo car,std::vector<cv::Point> roiContour)
{
    // 创建两个示例轮廓（不规则目标）
    std::vector<cv::Point> carContour;
    // std::vector<cv::Point> roiContour;
    
    carContour.push_back(cv::Point(car.x1,car.y1));
    carContour.push_back(cv::Point(car.x2,car.y1));
    carContour.push_back(cv::Point(car.x2,car.y2));
    carContour.push_back(cv::Point(car.x1,car.y2));
    
    // for(int i = 0; i< jROI.size(); i++)
    // {
    //     roiContour.push_back(cv::Point(jROI[i][0].asInt(),jROI[i][1].asInt()));
    // }
    // 创建轮廓集合
    std::vector<std::vector<cv::Point>> contours = { carContour, roiContour };

    // 计算轮廓的凸包
    std::vector<std::vector<cv::Point>> convexHulls;
    for (const auto& contour : contours) {
        std::vector<cv::Point> convexHull;
        cv::convexHull(contour, convexHull);
        convexHulls.push_back(convexHull);
    }

    // 判断两个不规则目标是否相交
    bool hasIntersection = false;
    for (size_t i = 0; i < convexHulls.size(); ++i) {
        for (size_t j = i + 1; j < convexHulls.size(); ++j) {
            if (cv::intersectConvexConvex(convexHulls[i], convexHulls[j], cv::noArray())) {
                hasIntersection = true;
                break;
            }
        }
        if (hasIntersection) {
            break;
        }
    }

    return hasIntersection;
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



static float GET_IOU(int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2)
{
    float lx = x1 > x2 ? x1 : x2;
    float ly = y1 > y2 ? y1 : y2;
    float rx = (x1 + w1) < (x2 + w2) ? (x1 + w1) : (x2 + w2);
    float ry = (y1 + h1) < (y2 + h2) ? (y1 + h1) : (y2 + h2);

    // if(lx > rx || rx > ry) return 0;

    float area_m = (rx - lx) * (ry - ly);

    float iou = area_m / ((float)(w1 * h1) + (float)(w2 * h2) - area_m);
    return iou;
}
static float GET_IOU2(int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2)
{
    float lx = x1 > x2 ? x1 : x2;
    float ly = y1 > y2 ? y1 : y2;
    float rx = (x1 + w1) < (x2 + w2) ? (x1 + w1) : (x2 + w2);
    float ry = (y1 + h1) < (y2 + h2) ? (y1 + h1) : (y2 + h2);

    // if(lx > rx || rx > ry) return 0;

    if(lx > rx || ly > ry)
        return false;
    return true;
}

static float GET_IOU_V2(int x1, int y1, int w1, int h1, int x2, int y2, int w2, int h2)//交集占目标2的iou值
{
    float lx = x1 > x2 ? x1 : x2;
    float ly = y1 > y2 ? y1 : y2;
    float rx = (x1 + w1) < (x2 + w2) ? (x1 + w1) : (x2 + w2);
    float ry = (y1 + h1) < (y2 + h2) ? (y1 + h1) : (y2 + h2);

    // if(lx > rx || rx > ry) return 0;

    float area_m = (rx - lx) * (ry - ly);

    float iou = area_m / ((float)(w2 * h2));
    return iou;
}


//置信度过滤
static void NMS_Filter(std::vector<BoxInfo> &detectResults, float nms_t)
{
    int ctime = detectResults.size();
    while (ctime--)
    {
        for (vector<BoxInfo>::iterator it1 = detectResults.begin(); it1 != detectResults.end(); it1++)
        {
            vector<BoxInfo>::iterator it2 = detectResults.begin();
            for (; it2 != detectResults.end(); it2++)
            {
                if (it1 == it2)
                    continue;
                float iou = GET_IOU(
                    it1->x1,
                    it1->y1,
                    it1->x1 - it1->x2,
                    it1->y1 - it1->y2,
                    
                    it2->x1,
                    it2->y1,
                    it2->x1 - it2->x2,
                    it2->y1 - it2->y2);
                // std::cout<<"iou:"<<iou<<std::endl;
                if (iou > nms_t)
                {
                    if (it1->score > it2->score)
                        detectResults.erase(it2);
                    else
                        detectResults.erase(it1);
                    break;
                }
            }
            if (it2 != detectResults.end())
                break;
        }
    }
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

static void RoiFillter2(vector<BoxInfo> &detectResults, std::vector<cv::Point> contours,std::string algsName)
{
    // std::string strs = jROI.toStyledString();
    // std::cout<<strs<<std::endl;
    if (contours.size() == 0)
        return;

    // std::vector<int>::const_iterator iter = b_list->begin();iter!=b_list->end();iter++
    int t = detectResults.size();

    for (int i = 0; i < t; i++) //遍历原有的框框图，把符合roi的标准筛选出来
    {
        for (std::vector<BoxInfo>::const_iterator r_ = detectResults.begin(); r_ != detectResults.end(); r_++)
        {
            bool isInRoi = false;
            if(algsName == "intrusion_detection" || 
            algsName == "over_the_wall_detection" || 
            algsName == "cv_perimeter_invasion" || 
            algsName == "climbing_recognition"
            )//
            {
                int xc = (r_->x1+r_->x2)/2;
                // int yc09 = r_->y2 - ((r_->y2-r_->y1)/10);
                int yc09 = r_->y2;
                cv::Point centPointOfBottom(xc,yc09);
                isInRoi = (cv::pointPolygonTest(contours, centPointOfBottom, true) < 0);
            }
            else
            {
                cv::Point CenterPoint((r_->x1 + r_->x2) / 2, (r_->y1 + r_->y2) / 2);
                isInRoi = (cv::pointPolygonTest(contours, CenterPoint, true) < 0);
            }

            if (isInRoi) //用于测试一个点是否在多边形中，contours是上面输入roi
            {
                detectResults.erase(r_);
                break;
            }
        }
    }
}


static void RoiFillter(vector<BoxInfo> &detectResults, Json::Value jROI,std::string &algsName)
{
    // std::string strs = jROI.toStyledString();
    // std::cout<<strs<<std::endl;
    if (jROI.size() == 0)
        return;

    std::vector<cv::Point> contours;

    for (int i = 0; i < jROI.size(); i++)
        contours.push_back(cv::Point(jROI[i][0].asInt(), jROI[i][1].asInt()));//把json数据的第 0 1 数据组装成 cv::point 放到 contours
    // std::vector<int>::const_iterator iter = b_list->begin();iter!=b_list->end();iter++
    int t = detectResults.size();

    for (int i = 0; i < t; i++) //遍历原有的框框图，把符合roi的标准筛选出来
    {
        for (std::vector<BoxInfo>::const_iterator r_ = detectResults.begin(); r_ != detectResults.end(); r_++)
        {
            bool isInRoi = false;
            if(algsName == "intrusion_detection" || 
            algsName == "over_the_wall_detection" || 
            algsName == "cv_perimeter_invasion" ||
            algsName == "climbing_recognition"
            )//
            {
                int xc = (r_->x1+r_->x2)/2;
                // int yc09 = r_->y2 - ((r_->y2-r_->y1)/10);
                int yc09 = r_->y2 ;
                cv::Point centPointOfBottom(xc,yc09);
                isInRoi = (cv::pointPolygonTest(contours, centPointOfBottom, true) < 0);
            }
            else
            {
                cv::Point CenterPoint((r_->x1 + r_->x2) / 2, (r_->y1 + r_->y2) / 2);
                isInRoi = (cv::pointPolygonTest(contours, CenterPoint, true) < 0);
            }

            if (isInRoi) //用于测试一个点是否在多边形中，contours是上面输入roi
            {
                detectResults.erase(r_);
                break;
            }
        }
    }
}





static bool StatusIntTime(std::string beginT, std::string endT)
{
    time_t rawtime;
    struct tm *info;
    time(&rawtime);

    info = localtime(&rawtime);
    int shi = info->tm_hour;
    int fen = info->tm_min;
	// cout<<shi<<":"<<fen<<endl;

    int timeGetter[4] = {0};
    int index = beginT.find(":");

    timeGetter[0] = atoi(beginT.substr(0, index).c_str());
    timeGetter[1] = atoi(beginT.substr(index + 1, beginT.size()).c_str());

    timeGetter[2] = atoi(endT.substr(0, index).c_str());
    timeGetter[3] = atoi(endT.substr(index + 1, endT.size()).c_str());

    int Time_min[3] = {0};
    Time_min[0] = timeGetter[0] * 60 + timeGetter[1];
    Time_min[1] = timeGetter[2] * 60 + timeGetter[3];
    Time_min[2] = info->tm_hour * 60 + info->tm_min;

    if(Time_min[0] > Time_min[1]) //23:00 - 5:00 跨天情况
    {
        if(Time_min[2] > Time_min[0] || Time_min[2] < Time_min[1])
            return true;
    }
	else
	{
		if (Time_min[2] > Time_min[0] && Time_min[2] < Time_min[1])
		{
			// if(info->tm_min > timeGetter[1] && info->tm_min < timeGetter[3])
			return true;
		}
	}

    return false;
}

static bool statusInTimes(Json::Value timeArr)
{
    if(timeArr.size() <= 0)
        return false;
    if(timeArr[0].isString())
    {
        return StatusIntTime(timeArr[0].asString(),timeArr[1].asString());
    }
    else
    {
        bool isNotTime = false;
        for(int i = 0; i< timeArr.size(); i++)
        {
            if(StatusIntTime(timeArr[i][0].asString(), timeArr[i][1].asString()))
            {
                isNotTime = true;
                break;
            }
        }
        return isNotTime;
    }
    return false;
}

static void _resize_and_encodebase64(std::string &base64, cv::Mat img,bool isResize)
{
    cv::Mat dstimg;
    if(isResize)
    {
        int w = 1280, h = 0;
        
        h = img.rows * 1280 /img.cols;
        if(h%2)
            h++;
        cv::resize(img, dstimg, cv::Size(w, h), cv::INTER_AREA);
    }
    else
    {
        dstimg = img;
    }

    
    cv_encode_base64_compression(dstimg, base64);
}

static std::string getMostBerthName(std::vector<std::string>& berthName)
{   
    std::string ans;
    std::map<std::string, int> tmp;
    for(auto item:berthName)
    {
        auto it = tmp.find(item);
        if(it == tmp.end())
            tmp[item] = 1;
        else
            tmp[item]++;
    }
    int maxIndex = -1;
    for(auto it = tmp.begin(); it!=tmp.end(); it++)
    {
        if(maxIndex < it->second)
        {
            maxIndex = it->second;
            ans = it->first;
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


static int isStopInRoi(std::vector<std::vector<cv::Point>> rois, SORT::TrackingBox carBox)
{
    int index = -1;
    if(rois.size() == 0)
        return index;
    cv::Point centerP(carBox.box.x + carBox.box.width/2, carBox.box.y + carBox.box.height/2);
    for(int i = 0; i< rois.size(); i++)
    {
        if(cv::pointPolygonTest(rois[i], centerP, true) >= 0)
        {
            return i;
        }
    }

    return index;
}


//防止拔了别人的车牌
static void check_car_plate_in_bottom(SORT::TrackingBox& box, BoxInfo& carPlateBox, std::string &carPlate)
{
    if(carPlate.size())
    {
        if((box.box.y + box.box.height/2) > carPlateBox.y1)
        {
            std::cout<<"[warning] this carPalte is On the Top of carBox ："<<carPlate<<std::endl;
            carPlate = "";
            carPlateBox.x1 = 0;
            carPlateBox.x2 = 0;
            carPlateBox.y1 = 0;
            carPlateBox.y2 = 0;
        }
    }
}

static bool check_car_is_charge(std::vector<BoxInfo> chargeGuns,SORT::TrackingBox carBox)
{
    std::vector<cv::Point> roi;
    roi.push_back(cv::Point(carBox.box.x, carBox.box.y));
    roi.push_back(cv::Point(carBox.box.x + carBox.box.width,carBox.box.y));
    roi.push_back(cv::Point(carBox.box.x + carBox.box.width,carBox.box.y + carBox.box.height));
    roi.push_back(cv::Point(carBox.box.x, carBox.box.y + carBox.box.height));

    for(auto x : chargeGuns)
    {
        if(cv::pointPolygonTest(roi, cv::Point(x.x1,x.y1), true) >= 0)
            return true;
        if(cv::pointPolygonTest(roi, cv::Point(x.x2,x.y1), true) >= 0)
            return true;
        if(cv::pointPolygonTest(roi, cv::Point(x.x1,x.y2), true) >= 0)
            return true;
        if(cv::pointPolygonTest(roi, cv::Point(x.x2,x.y2), true) >= 0)
            return true;
    }
    return false;
}

static bool get_aver_charge(std::vector<int> vec)
{
    int zero = 0,notZero = 0;
    for(auto x : vec)
    {
        if(x)
            notZero++;
        else
            zero++;
    }
    return zero > notZero ? false:true;
}

static void check_green_bigger_than_2(std::vector<std::string> &vec)
{
    int cnt = 0;
    for(auto item:vec)
    {
        if(item.size() == 10)
            cnt++;
    }
    if(cnt<0)
    {
        //do nothing
    }
    else//大于2，把油车车牌清零
    {
        REdel:
        for(auto it = vec.begin(); it!= vec.end(); it++)
        {
            if(it->size()<10)
            {
                vec.erase(it);
                goto REdel;
            }
        }
    }
}



static int chooseBestBoxInfoIndex(std::vector<BoxInfo>& vec)
{
    int ans = -1;
    float maxScore = 0;
    for(int i = 0; i < vec.size(); i++)
    {
        if(vec[i].score > maxScore)
        {
            maxScore = vec[i].score;
            ans = i;
        }
    }

    return ans;
}


static BoxInfo traceBox2BoxInfo(SORT::TrackingBox box)
{
    BoxInfo ans;
    ans.x1 = box.box.x;
    ans.y1 = box.box.y;
    ans.x2 = box.box.x + box.box.width;
    ans.y2 = box.box.y + box.box.height;
    ans.label = box.label;
    ans.score = box.score;
    return ans;
}









static bool no_ID_fire_smoking_detect(BoxInfo& oldBox, BoxInfo boxNew, int limArea)
{
    if(oldBox.x1 == 0 && oldBox.x2 == 0)
    {
        oldBox.x1 =boxNew.x1;
        oldBox.x2 =boxNew.x2;
        oldBox.y1 =boxNew.y1;
        oldBox.y2 =boxNew.y2;
        return false;
    }
    float height1 = oldBox.y2 - oldBox.y1;
    float height2 = boxNew.y2 - boxNew.y1;
    float area =  (float)std::abs(boxNew.y2 - boxNew.y1)*std::abs(boxNew.x2 - boxNew.x1);
    float drift = (float)std::abs(height1 - height2)/height2;

    if(drift>=0.05 && area >= (limArea*limArea)) 
    {
        oldBox.x1 =boxNew.x1;
        oldBox.x2 =boxNew.x2;
        oldBox.y1 =boxNew.y1;
        oldBox.y2 =boxNew.y2;
        return true;
    }
    return false;
}

static bool same_place_detect(BoxInfo& oldBox, BoxInfo boxNew)
{
    if(oldBox.x1 == 0 && oldBox.x2 == 0)
    {
        oldBox.x1 =boxNew.x1;
        oldBox.x2 =boxNew.x2;
        oldBox.y1 =boxNew.y1;
        oldBox.y2 =boxNew.y2;
        return false;
    }
    bool isDiff = (std::abs(boxNew.x1 - oldBox.x1) > 100 || std::abs(boxNew.x2 - oldBox.x2) > 100|| std::abs(boxNew.y1 - oldBox.y1) > 100|| std::abs(boxNew.y2 - oldBox.y2) > 100);
    if(isDiff)
        return true;
    else
        return false;
}

static bool same_place_detect_arr(std::vector<BoxInfo> &oldBoxs, BoxInfo &boxNew)
{
    if(oldBoxs.size() == 0)
        return false;
    bool isSamePlace = false;
    for(auto x:oldBoxs)
    {
        int isDiff = (std::abs(boxNew.x1 - x.x1) + std::abs(boxNew.x2 - x.x2) + std::abs(boxNew.y1 - x.y1)+ std::abs(boxNew.y2 - x.y2));
        // printf("[warn] %d\n",isDiff);
        if(isDiff < 150)
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
    if(oldBoxs.size() == 0)
        return false;
    bool isSamePlace = false;
    for(auto x:oldBoxs)
    {
        int isDiff = (std::abs(boxNew.box.x - x.box.x) + std::abs(boxNew.box.y - x.box.y) + std::abs(boxNew.box.width - x.box.width)+ std::abs(boxNew.box.height - x.box.height));
        // printf("[warn] %d\n",isDiff);
        if(isDiff < 150)
        {
            // printf("[warn] isSamePlace!\n");
            isSamePlace = true;
            break;
        }
    }
    return isSamePlace;
}

// static bool no_ID_mouse_detect(BoxInfo& oldBox, BoxInfo boxNew, int limArea)
// {
//     if(oldBox.x1 == 0 && oldBox.x2 == 0)
//     {
//         oldBox.x1 =boxNew.x1;
//         oldBox.x2 =boxNew.x2;
//         oldBox.y1 =boxNew.y1;
//         oldBox.y2 =boxNew.y2;
//         return false;
//     }
//     float height1 = oldBox.y2 - oldBox.y1;
//     float height2 = boxNew.y2 - boxNew.y1;
//     float area =  (float)std::abs(boxNew.y2 - boxNew.y1)*std::abs(boxNew.x2 - boxNew.x1);
//     float drift = (float)std::abs(height1 - height2)/height2;

//     if(drift>=0.05 && area >= (limArea*limArea)) 
//     {
//         oldBox.x1 =boxNew.x1;
//         oldBox.x2 =boxNew.x2;
//         oldBox.y1 =boxNew.y1;
//         oldBox.y2 =boxNew.y2;
//         return true;
//     }
//     return false;
// }
static void getGiggerOutSizeReat2(int &x, int &y, int &w, int &h, Json::Value roi1, Json::Value roi2)
{
    int maxX = -1, minX = 9999, maxY = -1, minY = 9999;
    for(int i = 0; i < roi1.size(); i++)
    {
        int tempx = roi1[i][0].asInt();
        if(tempx >  maxX)
            maxX = tempx;
        if(tempx <  minX)
            minX = tempx;
        tempx = roi1[i][1].asInt();
        if(tempx >  maxY)
            maxY = tempx;
        if(tempx <  minY)
            minY = tempx;
    }
    for(int i = 0; i < roi2.size(); i++)
    {
        int tempx = roi2[i][0].asInt();
        if(tempx >  maxX)
            maxX = tempx;
        if(tempx <  minX)
            minX = tempx;
        tempx = roi2[i][1].asInt();
        if(tempx >  maxY)
            maxY = tempx;
        if(tempx <  minY)
            minY = tempx;
    }
    x = minX;
    y = minY;
    w = maxX - minX;
    h = maxY - minY;
}


static bool leave_post_detect_(vector<BoxInfo> &generate_boxes, int limCnt)
{ 
    int bodyCnt = 0;
    int headCnt = 0;
    for(int i =0; i < generate_boxes.size(); i++)
    {
        if(generate_boxes[i].label == 0)
            bodyCnt++;
        else
            headCnt++;
    }
    return !((bodyCnt >= limCnt) || (headCnt >= limCnt));
}


// static bool smocking_detect(vector<BoxInfo> box1,vector<BoxInfo> box2)
// {
//     for(auto item:box1)
//     {
//         for(auto item2:box2)
//         {
//             int h1 = item.y2 - item.y1;
//             int w1 = item.x2 - item.x1;

//             int h2 = item2.y2 - item2.y1;
//             int w2 = item2.x2 - item2.x1;
//             float areaRatio = abs(float(h1*w1)/float(h2*w2)); //面积大小
//             if(areaRatio > 0.5)
//                 return false;
//             if(0.001 < GET_IOU2(item.x1,item.y1,w1,h1,item2.x1,item2.y1,w2,h2)) // 交集
//             {
//                 // std::cout<<"iou :"<<GET_IOU2(item.x1,item.y1,w1,h1,item2.x1,item2.y1,w2,h2)<<"\n";
//                 return true;
//             }
//         }
//     }
//     return false;
// }

static bool smocking_detect2(vector<BoxInfo> box0,vector<BoxInfo> box1) //0吸烟1人2头
{
    for(auto item:box0)
    {
        if (item.label==0){
            for(auto item2:box1)
            {
                if (item2.label==2 && item2.score>0.4){
                    int h1 = item.y2 - item.y1;
                    int w1 = item.x2 - item.x1;

                    int h2 = item2.y2 - item2.y1;
                    int w2 = item2.x2 - item2.x1;
                    float areaRatio = abs(float(h1*w1)/float(h2*w2)); //面积大小
                    if(areaRatio > 0.8)
                        return false;
                    // if(0.001 < GET_IOU2(item.x1,item.y1,w1,h1,item2.x1,item2.y1,w2,h2)) // 交集
                    if(true == GET_IOU2(item.x1,item.y1,w1,h1,item2.x1,item2.y1,w2,h2)) // 交集
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

static int garbage_detect2(vector<BoxInfo> box1,vector<BoxInfo> box2,vector<BoxInfo> &box3)
{//0 无目标 1 报警 2有目标但有交集
    if(box1.size()<= 0)
        return 0;

    bool isIoU = true;
    for(auto item:box1)
    {
        bool itemIou = true;
        for(auto item2:box2)
        {
            if (item2.label==4 && item2.score>0.4){    //0123 垃圾 4人5头
            int h1 = item.y2 - item.y1;
            int w1 = item.x2 - item.x1;

            int h2 = item2.y2 - item2.y1;
            int w2 = item2.x2 - item2.x1;
            // float areaRatio = abs(float(h1*w1)/float(h2*w2)); //面积大小
            // if(areaRatio > 0.5)
            //     return false;
            if(0.01 < GET_IOU2(item.x1,item.y1,w1,h1,item2.x1,item2.y1,w2,h2)) // 有交集
            {
                // std::cout<<"iou :"<<GET_IOU2(item.x1,item.y1,w1,h1,item2.x1,item2.y1,w2,h2)<<"\n";
                itemIou = false;
                break;
            }
            }
        }
        if(itemIou)
        {
            box3.push_back(item);
            isIoU = false;
        }
            
    }
    if(!isIoU)
        return 1;
    else
        return 2;
}



static  bool in_roi_check_(std::vector<cv::Point> contours,CarBoxInfo objs,float pro)
{

    // for(int i =0; i< 4; i++)
    // {
    //     cv::Point CenterPoint(objs.landmark[2*i] , objs.landmark[2*i+1]);

    //     if(cv::pointPolygonTest(contours, CenterPoint, true) > 0)
    //     {
    //         return true;
    //     }
    // }
    // return false;

    std::vector<cv::Point> carContours;
    for(int i =0; i< 4; i++)
    {
        carContours.push_back(cv::Point (objs.landmark[2*i] , objs.landmark[2*i+1]));
    }
    std::vector<cv::Point> convexHull1, convexHull2;
    cv::convexHull(contours, convexHull1);
    cv::convexHull(carContours, convexHull2);
    // 计算交集
    std::vector<cv::Point> intersectionPoints;
    cv::intersectConvexConvex(convexHull1, convexHull2, intersectionPoints);
    if(intersectionPoints.size() == 0)
        return false;
    // 计算交集的面积
    double intersectionArea = cv::contourArea(intersectionPoints);
    //计算交集面积于车面积的占比
    double proportion = intersectionArea/cv::contourArea(carContours);
    // std::cout<<"[INFO] proportion:"<<proportion<<"\n";
    
    return proportion > pro ? true:false;   

}



static void RoiFillterObj(vector<CarBoxInfo> &detectResults, Json::Value jROI)
{
    // std::string strs = jROI.toStyledString();
    // std::cout<<strs<<std::endl;
    if (jROI.size() == 0)
        return;

    std::vector<std::vector<cv::Point>> contourss;

    for (int i = 0; i < jROI.size(); i++)
    {
        std::vector<cv::Point> contours;
        for(int j = 0; j < jROI[i].size(); j++)
            contours.push_back(cv::Point(jROI[i][j][0].asInt(), jROI[i][j][1].asInt()));
        contourss.push_back(contours);
    }
        

    ReDelItem:
    for (int i = 0; i < detectResults.size(); i++) //遍历原有的框框图，把符合roi的标准筛选出来
    {
        bool isInRois = false;
        for(int j = 0; j < contourss.size(); j++)
        {
            cv::Point CenterPoint((detectResults[i].x1 + detectResults[i].x2)/2 , (detectResults[i].y1 + detectResults[i].y2)/2);
            if (cv::pointPolygonTest(contourss[j], CenterPoint, true) >= 0) //用于测试一个点是否在多边形中，contours是上面输入roi
            {
                isInRois = true;
                break;
            }
        }
        if(!isInRois)
        {
            detectResults.erase(detectResults.begin()+i);
            goto ReDelItem;
        }
    }
} 

static void CarBoxInfo2BoxInfo(vector<CarBoxInfo>& carboxes, vector<BoxInfo>& boxes)
{
    for(auto x : carboxes)
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