
// 0断线/1在线/2异常（花屏黑屏）/3遮挡/4偏离/5在线且正常

#include "post/model_base.h"
#include <thread>
#include <vector>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <thread>
#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <map>
#include <ctime>
#include <memory>
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/features2d.hpp>
#include <bits/stdint-uintn.h>
#include "common/http/httplib.h"
#include "common/ConvertImage.h"
#include "prefer/get_videoffmpeg.hpp" 
#include "license.h"
#include "common/json/json.hpp"
#include "post_util.h"
#include "httpCamStatus.h"

using json = nlohmann::json;

string ip_port = "127.0.0.1:9696";

// 判断遮挡：亮度方差法
bool isOccluded(const cv::Mat& frame) {
    if (frame.empty()) return true;
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    // 方差小于阈值，说明画面变化很小，可能被遮挡
    return stddev[0] < 10; // 阈值可调
}

// 判断遮挡：边缘检测法
bool isOccludedByEdge(const cv::Mat& frame, int edge_thresh = 500) {
    if (frame.empty()) return true;
    cv::Mat gray, edges;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    else
        gray = frame;
    // Canny边缘检测
    cv::Canny(gray, edges, 50, 150);
    int edge_count = cv::countNonZero(edges);
    // 边缘像素点数量小于阈值，认为被遮挡
    return edge_count < edge_thresh;
}

// 判断花屏：均值和方差法 + 单色块检测
bool isCorrupted(const cv::Mat& frame) {
    if (frame.empty()) return true;
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);

    // 1. 全黑/全白/全灰
    if (mean[0] < 10 || mean[0] > 245 || stddev[0] < 5) return true;

    // 2. 单色块检测
    int histSize = 256;
    float range[] = {0, 256};
    const float* histRange = {range};
    cv::Mat hist;
    cv::calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange);
    double maxVal = 0;
    cv::minMaxLoc(hist, 0, &maxVal, 0, 0);
    if (maxVal > gray.total() * 0.8) return true; // 超过80%像素为同一灰度

    return false;
}

static bool cameraIsDity(const cv::Mat& im)
{
    cv::Mat img;
    cv::cvtColor(im,img, cv::COLOR_BGR2GRAY);
    cv::Mat sobelx;
    cv::Sobel(img,sobelx, 6, 1, 0,3);
    cv::Mat sobely;
    cv::Sobel(img, sobely,6, 0, 1,3);
    cv::Mat edge;
    cv::addWeighted(sobelx, 0.5, sobely, 0.5, 0,edge);
    cv::Scalar aa=cv::sum(cv::abs(edge));
    double bb = (double)aa[0]/(double)(im.cols*im.rows);
    std::cout<<"[INFO] Dity14:"<<bb<<"\n";
    img.release();
    sobely.release();
    sobelx.release();
    // std::cout<<(double)aa[1]<<"\n";
    if (bb>14)
    {
        return false;
    }
    else
        return true;
}

// 判断偏离：ORB特征点匹配法
bool isDeviatedByFeature(const cv::Mat& ref_frame, const cv::Mat& frame, int min_match_count = 20) {
    if (ref_frame.empty() || frame.empty()) return false;

    cv::Mat gray_ref, gray_cur;
    if (ref_frame.channels() == 3)
        cv::cvtColor(ref_frame, gray_ref, cv::COLOR_BGR2GRAY);
    else
        gray_ref = ref_frame;
    if (frame.channels() == 3)
        cv::cvtColor(frame, gray_cur, cv::COLOR_BGR2GRAY);
    else
        gray_cur = frame;

    // ORB特征提取
    auto orb = cv::ORB::create();
    std::vector<cv::KeyPoint> kp_ref, kp_cur;
    cv::Mat desc_ref, desc_cur;
    orb->detectAndCompute(gray_ref, cv::Mat(), kp_ref, desc_ref);
    orb->detectAndCompute(gray_cur, cv::Mat(), kp_cur, desc_cur);

    if (desc_ref.empty() || desc_cur.empty()) return true; // 没有特征点，视为偏离

    // BFMatcher暴力匹配
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(desc_ref, desc_cur, matches);

    // 统计好匹配数量
    int good_matches = 0;
    for (const auto& m : matches) {
        if (m.distance < 50) good_matches++; // 距离阈值可调
    }
    std::cout<<"[INFO] offset_angle:"<<good_matches<<" "<<min_match_count<<"\n";

    // 匹配点数低于阈值，认为偏离
    return good_matches < min_match_count;
}


int main(int argc, char **argv)
{
    string flag = "0"; // 0有延迟默认开启，1有参数是给美亚的
    if (argc == 2){
        flag = argv[1];
    }
    #ifndef DEED_DEBUG
    if (!checklic("../civiapp/licens/license.lic"))
    {
        std::cout << "[Error] Check license failed!" << std::endl;
        return 0;
    }
    #endif
    map<string, cv::Mat> ref_frames; // 用于保存每个摄像头的基准帧       
    json cam_list;
    int offset_angle = 20; // 偏移角度，小不容易误报
    ImagemConverter m2s;
    HttpCamStatus aa; //接口
    while (true)
    {
        if(flag=="0"){
            sleep(10);
            continue; 
        }
        if (auto res = httplib::Client(ip_port.c_str()).Post("/api/camera/list"))
        {
            auto data = res->body;
            cam_list = json::parse(data)["data"];
        }
        if (auto res = httplib::Client(ip_port.c_str()).Post("/api/camera/offset_angle"))
        {
            auto data = res->body;
            offset_angle = json::parse(data)["data"]["cam_offset_angle"];
            
        }
        // TODO: update_camera_status 不要一直发送，状态有变化，持续时间和间隔时间一起控制
        for (const auto& cam : cam_list)
        {
            string rtsp_path= cam["rtsp"];
            std::cout << "Camera Name: " << cam["rtsp"] << std::endl;
            // 1. 检查摄像头离线
            VideoDeal capture(rtsp_path, 0);
            if (capture.status){
                update_camera_status(rtsp_path, 1, "");
                std::cout << "[INFO] cam online: " << rtsp_path << '\n';     
            }  
            else
            {
                update_camera_status(rtsp_path, 0, "open cam error.");
                std::cerr << "[ERROR] cam open error: " << rtsp_path << '\n';       
                continue;   
            }

            // 2. 检查摄像头花屏
            cv::Mat frame;
            for(int i = 0; i < 20; ++i) { // 尝试获取20次帧
                int getRet = capture.getframe(frame);
                if (getRet == 0) break; // 成功获取帧
                std::this_thread::sleep_for(std::chrono::milliseconds(50)); // 等待200ms再试
            }
            if(frame.empty())
            {
                update_camera_status(rtsp_path, 2, "get frame error.","");
                std::cerr << "[ERROR] cam get frame error: " << rtsp_path << '\n';
                continue;
            }
            string img64 = m2s.mat2str(frame);
            if (isCorrupted(frame)) {
                update_camera_status(rtsp_path, 2, "frame corrupted", img64);
                std::cerr << "[ERROR] cam corrupted: " << rtsp_path << '\n';
                continue;
            }

            // 3. 检查摄像头遮挡
            if (isOccluded(frame) || isOccludedByEdge(frame) || cameraIsDity(frame)) {
                update_camera_status(rtsp_path, 3, "camera occluded", img64);
                std::cerr << "[ERROR] cam occluded: " << rtsp_path << '\n';
                continue;
            }

            // 4. 检查摄像头偏离
            if (ref_frames[rtsp_path].empty()) {
                ref_frames[rtsp_path] = frame.clone(); // 初始化基准帧
            } else {
                if (isDeviatedByFeature(ref_frames[rtsp_path], frame,offset_angle)) {
                    update_camera_status(rtsp_path, 4, "camera deviated", img64);
                    std::cerr << "[ERROR] cam deviated: " << rtsp_path << '\n';
                    ref_frames[rtsp_path] = frame.clone();
                    continue;
                }
            }

            update_camera_status(rtsp_path, 5, "camera ok", "");
            usleep(100*1000);
        }
        sleep(3);
        cout<<"***********************************-------------------------------"<<endl;
    }

    return 1;
}
// 内存占用会涨到1G，不会再往上涨