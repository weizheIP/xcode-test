#ifndef SORT_H
#define SORT_H

#include <vector>
// #include <opencv2/opencv.hpp>
// #include <opencv2/video/tracking.hpp>


#include "KalmanTracker.h"

//using namespace std;

typedef struct Kal_Label_Struct{
    KalmanTracker kalmT;
    float score;
    int label;
};

class SORT
{
private:

public:
    typedef struct TrackingBox{
        int frame;
        int id;
        int label;
        float score;
        cv::Rect_<float> box;
    }TrackingBox;

    SORT(){;}
    // std::vector<KalmanTracker> trackers;
    std::vector<Kal_Label_Struct> trackers;

    std::vector<std::vector<TrackingBox>> detFrameData;

    // std::vector<TrackingBox> Sortx(std::vector<std::vector<float>> bbox, int fi, std::string sTypeName="");
    std::vector<TrackingBox> Sortx(std::vector<std::vector<float>> bbox, int fi);
    double GetIOU(cv::Rect_<float> bb_dr, cv::Rect_<float> bb_gt);
    double GetDIOU(cv::Rect_<float> bb_dr, cv::Rect_<float> bb_gt);

    // KalmanTracker kalman;
    int sort_kal_index = 0;

};

void cal_count2part(std::vector<SORT::TrackingBox> curframe, std::string dd, int line, std::vector<int>* b_list, std::vector<int>* t_list, int* count,int &area);
void cal_count2part(std::vector<SORT::TrackingBox> curframe, std::string dd, int line, std::vector<int>* b_list, std::vector<int>* t_list, int* count ,int* count_add,int* count_sub,int &area);

#endif


