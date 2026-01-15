#pragma once

#include "post_util.h"

static float calculateAngle(BoxInfo center, BoxInfo point, json setting_value) // 获取表盘角度值，（°）
{
    cv::Point center_c, point_c;
    center_c.x = (center.x1 + (center.x2 - center.x1) / 2);
    center_c.y = (center.y1 + (center.y2 - center.y1) / 2);
    point_c.x = (point.x1 + (point.x2 - point.x1) / 2);
    point_c.y = (point.y1 + (point.y2 - point.y1) / 2);

    // GPT: 在C++中，要判断一个点相对于原点的连线与y轴负方向顺时针之间的角度，y轴负方向为0度，顺时针一圈，y周正方向为180度，x轴正方向为270度
    double angleFromXAxis = atan2(-(point_c.y - center_c.y), point_c.x - center_c.x);
    // Convert angle to degrees
    double angleDeg = angleFromXAxis * (180.0 / M_PI); // x轴正方向
    if (angleDeg < 0)
    {
        angleDeg += 360.0;
    }
    angleDeg = 360 - 90 - angleDeg; // y轴负方向为0度
    if (angleDeg < 0)
    {
        angleDeg += 360.0;
    }

    // 转换值
    float Min_V = setting_value["Min_Value"];
    float Max_V = setting_value["Max_Value"];
    float Min_A = setting_value["Min_Angle"];
    float Max_A = setting_value["Max_Angle"]; // 垂直向下为0度，一般是45度到315度
    float ValueBuff = Max_V - Min_V;
    float AngleBuff = Max_A - Min_A;

    // float value_now = ValueBuff / AngleBuff * (angleDeg - Min_A);
    float value_now = (ValueBuff / AngleBuff * (angleDeg - Min_A) )+ Min_V; //最小不一定是0
    return value_now;
}

static vector<float> calMechanical(std::vector<BoxInfo> output, std::vector<BoxInfo>& output_big, vector<vector<vector<int>>> Roi, json setting_values) // 获取表盘角度值，（°）
{
    std::vector<std::vector<cv::Point>> contourss;
    for (int i = 0; i < Roi.size(); i++)
    {
        std::vector<cv::Point> contours;
        for (int j = 0; j < Roi[i].size(); j++)
            contours.push_back(cv::Point(Roi[i][j][0], Roi[i][j][1]));
        contourss.push_back(contours);
    }
    std::vector<std::vector<BoxInfo>> output_new;
    for (int i = 0; i < contourss.size(); i++)
    {
        BoxInfo d_tmp;
        std::vector<BoxInfo> detection3;
        for (auto detection : output)
        {
            if (cv::pointPolygonTest(contourss[i], cv::Point((detection.x1+detection.x2)/2, (detection.y1+detection.y2)/2), true) > 0)
            {
                detection3.push_back(detection);
                if(detection.label == 2)
                    d_tmp=detection;
            }
        }
        output_new.push_back(detection3);
        output_big.push_back(d_tmp);
    }
    int j=0;    
    vector<float> out;    
    for (auto detection3 : output_new)
    {
        bool flag1=false;
        bool flag2=false;
        BoxInfo center, point;
        for (auto detection1 : detection3)
        {
            if (detection1.label == 1){
                center = detection1;
                flag1=true;
            }
            else if (detection1.label == 0){
                point = detection1;
                flag2=true;
            }
        }
        if(flag1==false or flag2==false)
            out.push_back(0);
        else{
            float value_now  =calculateAngle(center, point, setting_values[j]);
            out.push_back(value_now);
        }
        j++;
    }
    return out;
}