#include "SORT.h"
#include "Hungarian.h"
#include "KalmanTracker.h"
#include <opencv2/opencv.hpp>

#include <set>
#include <vector>
#include <chrono>




extern string sTypeName;
double SORT::GetIOU(cv::Rect_<float> bb_dr, cv::Rect_<float> bb_gt){
    float in = (bb_dr & bb_gt).area();
    float un = bb_dr.area() + bb_gt.area() - in;

    if(un < DBL_EPSILON)
        return 0;

    double iou = in / un;

    return iou;
}

// #if 1
double SORT::GetDIOU(cv::Rect_<float> bb_dr, cv::Rect_<float> bb_gt)
{
    double iou = GetIOU(bb_dr,bb_gt);
    // 两矩形中心的距离
    double center_dis = std::sqrt(std::pow(bb_dr.x + bb_dr.width/2 - bb_gt.x - bb_gt.width/2,2) + 
    std::pow(bb_dr.y + bb_dr.height/2 - bb_gt.y - bb_gt.height/2,2));
    // 两矩形最小外接矩形左上-右下距离
    std::vector<float> x_list{bb_dr.x,
        bb_dr.x + bb_dr.width,
        bb_gt.x,
        bb_gt.x + bb_gt.width};
    std::vector<float> y_list{bb_dr.y,
        bb_dr.y + bb_dr.height,
        bb_gt.y,
        bb_gt.y + bb_gt.height};

    float x_max = *std::max_element(x_list.begin(),x_list.end());
    float y_max = *std::max_element(y_list.begin(),y_list.end());
    float x_min = *std::min_element(x_list.begin(),x_list.end());
    float y_min = *std::min_element(y_list.begin(),y_list.end());
    // x_max = std::max(np.maximum(np.maximum(bb_test[..., 0], bb_test[..., 2]),bb_gt[..., 0]), bb_gt[..., 2])
    // x_min = np.minimum(np.minimum(np.minimum(bb_test[..., 0], bb_test[..., 2]),bb_gt[..., 0]), bb_gt[..., 2])
    // y_max = np.maximum(np.maximum(np.maximum(bb_test[..., 1], bb_test[..., 3]),bb_gt[..., 1]), bb_gt[..., 3])
    // y_min = np.minimum(np.minimum(np.minimum(bb_test[..., 1], bb_test[..., 3]),bb_gt[..., 1]), bb_gt[..., 3])

    // #ifdef _FISH_TDDS
    // if(sTypeName == "cv_fish_tdds")
    // {
    float all_area = 0;
    float aUb_area = 0;
    if(iou <= 0) //无交集
    {
        aUb_area = bb_dr.area() + bb_gt.area();
    }
    else //有交集
    {
        aUb_area = bb_dr.area() + bb_gt.area() - (bb_dr & bb_gt).area();
    }
    all_area = (x_max - x_min ) * (y_max - y_min);
    
    float ans  = iou - ((all_area - aUb_area) / all_area);

    return ans;
    // }
    // // #else
    // else{
    // double outer_diagonal_line = std::sqrt(std::pow(x_max-x_min,2) + std::pow(y_max-y_min,2));
    // // std::cout<<"diou:"<<iou - center_dis / outer_diagonal_line<<std::endl;
    // if(iou - center_dis / outer_diagonal_line < -1){
    //     return -1;
    // }
    // return iou - center_dis / outer_diagonal_line;
    // }
    // #endif
}



// std::vector<SORT::TrackingBox> SORT::Sortx(std::vector<std::vector<float>> bbox, int fi, std::string sTypeName=""){
std::vector<SORT::TrackingBox> SORT::Sortx(std::vector<std::vector<float>> bbox, int fi){
    
    int min_hits = 0; //min time target appear
    // #ifdef TDDS
    // double iouThreshold = 0.1;//matching IOU
    // int max_age = 20;//max time object disappear
    // #elif _FISH_TDDS
    // int max_age = 3;//max time object disappear
    // double iouThreshold = -0.5;//matching IOU
    // #elif TRUNK_
    // min_hits = 2; //min time target appear
    // int max_age = 20;//max time object disappear
    // double iouThreshold = 0.1;//matching IOU
    // #else
    // int max_age = 5;//max time object disappear
    // double iouThreshold = 0.01;//matching IOU
    // #endif

    // int max_age = 20;iouThreshold = 0.1; //猪
    // double iouThreshold = 0.1;//matching IOU 猪的通道点数0.1，稳定点但是保证帧率够就行，0.05猪斜着或者挡住会有id一样的问题；人如果用0.1甩头就会变id，人用0.01。
    // cv_tdds_piggy用5，0.01跑的也挺好？
    // max_age = 20,左边有猪会记住，导致右边20帧又出现同一ID


    int max_age = 5;// person
    double iouThreshold = 0.01;// person
    if(sTypeName == "cv_fish_tdds")
    {
        max_age = 3;
        iouThreshold = -0.5;
    }else if(sTypeName == "tdds" || sTypeName == "tdgz" || sTypeName == "cv_tdgz"){
        iouThreshold = 0.1;
        max_age = 5;
    }else if(sTypeName == "cv_tdds_piggy"  || sTypeName == "cv_count_chicken"){
        iouThreshold = 0.06;
        max_age = 5;
    }else if(sTypeName == "cv_livestock_license_plates_detection" || sTypeName == "cv_large_car_detection"  || sTypeName == "cv_parking_license_plate_detection"  || sTypeName == "cv_road_parking_license_plate_detection" ){
        min_hits = 2; 
        max_age = 20;
        iouThreshold = 0.1;
    }

    // variables used in the sort-loop
    std::vector<SORT::TrackingBox> detData;
    std::vector<cv::Rect_<float>> predictedBoxes;
    std::vector<std::vector<double>> iouMatrix;
    std::vector<int> assignment;

    std::set<int> unmatchedDetections;
    std::set<int> unmatchedTrajectories;
    std::set<int> allItems;
    std::set<int> matchedItems;
    // result
    std::vector<cv::Point> matchedPairs;
    std::vector<SORT::TrackingBox> frameTrackingResult;

    unsigned int trkNum = 0;
    unsigned int detNum = 0;

    // time
    auto start_time = std::chrono::high_resolution_clock::now();

    // bounding boxes in a frame store in detFrameData
    for (int i = 0; i < bbox.size() ; i++){
        SORT::TrackingBox tb;
        tb.frame = fi + 1;
        tb.box = cv::Rect_<float>(cv::Point_<float>(bbox[i][0], bbox[i][1]), cv::Point_<float>(bbox[i][2], bbox[i][3]));
        // 增加跟踪目标label和score
        tb.score = bbox[i][4];
        tb.label = bbox[i][5];

        detData.push_back(tb);
    }

    detFrameData.push_back(detData);
    if(sort_kal_index>2147483600)sort_kal_index=0;
    // std::cout << "reading bbox from yolo \n";
    // initialize kalman trackers using first detections.
    if(trackers.size() == 0){
        std::vector<SORT::TrackingBox> first_frame;
        for (unsigned int i = 0; i < detFrameData[fi].size(); i++){
            
            KalmanTracker trk = KalmanTracker(detFrameData[fi][i].box, sort_kal_index);
            // 增加跟踪目标label和score
            Kal_Label_Struct kls={
                trk,detFrameData[fi][i].score,detFrameData[fi][i].label
            };
            trackers.push_back(kls);
            // trackers.push_back(trk);

        }
        // output the first frame detections
        // for (unsigned int id = 0; id < detFrameData[fi].size(); id++){
        //     SORT::TrackingBox tb = detFrameData[fi][id];
        //     tb.id = id +1 ;
        //     first_frame.push_back(tb);
        //     //std::cout << tb.frame << "," << id + 1 << "," << tb.box.x << "," << tb.box.y << "," << tb.box.width << "," << tb.box.height  << std::endl;
        // }
        return first_frame;
    }
    

    /*
    3.1. get predicted locations from existing trackers
    */
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        cv::Rect_<float> pBox = (*it).kalmT.predict();
        //std::cout << pBox.x << " " << pBox.y << std::endl;
        if (pBox.x >= 0 && pBox.y >= 0)
        {
            predictedBoxes.push_back(pBox);
            it++;
        }
        else
        {
            it = trackers.erase(it);
            //cerr << "Box invalid at frame: " << frame_count << endl;
        }
    }

    /*
    3.2. associate detections to tracked object (both represented as bounding boxes)
    */
    trkNum = predictedBoxes.size();
    detNum = detFrameData[fi].size();
    iouMatrix.resize(trkNum, std::vector<double>(detNum, 0));
    if(trkNum==0)
    {
        // std::cout<<"trknum is empty"<<std::endl;
        return frameTrackingResult;
    }

    for (unsigned int i = 0; i < trkNum; i++) // compute iou matrix as a distance matrix
    {
        for (unsigned int j = 0; j < detNum; j++)
        {
            // use 1-iou because the hungarian algorithm computes a minimum-cost assignment.
            // #ifdef _FISH_TDDS
            if(sTypeName == "cv_fish_tdds")
            {
            iouMatrix[i][j] = 1 - GetDIOU(predictedBoxes[i], detFrameData[fi][j].box);
            }
            // #else
            else{
            iouMatrix[i][j] = 1 - GetIOU(predictedBoxes[i], detFrameData[fi][j].box);
            }
            // #endif
        }
    }

    // solve the assignment problem using hungarian algorithm.
    // the resulting assignment is [track(prediction) : detection], with len=preNum
    HungarianAlgorithm HungAlgo;
    HungAlgo.Solve(iouMatrix, assignment);

    // find matches, unmatched_detections and unmatched_predictions
    if (detNum > trkNum) //	there are unmatched detections
    {
        for (unsigned int n = 0; n < detNum; n++)
            allItems.insert(n);

        for (unsigned int i = 0; i < trkNum; ++i)
            matchedItems.insert(assignment[i]);

        // calculate the difference between allItems and matchedItems, return to unmatchedDetections
        std::set_difference(allItems.begin(), allItems.end(),
            matchedItems.begin(), matchedItems.end(),
            insert_iterator<set<int>>(unmatchedDetections, unmatchedDetections.begin()));
    }
    else if (detNum < trkNum) // there are unmatched trajectory/predictions
    {
        for (unsigned int i = 0; i < trkNum; ++i)
            if (assignment[i] == -1) // unassigned label will be set as -1 in the assignment algorithm
                unmatchedTrajectories.insert(i);
    }
    else
        ;

    // filter out matched with low IOU
    // output matchedPairs
    for (unsigned int i = 0; i < trkNum; ++i)
    {
        if (assignment[i] == -1) // pass over invalid values
            continue;
        if (1 - iouMatrix[i][assignment[i]] < iouThreshold)
        {
            unmatchedTrajectories.insert(i);
            unmatchedDetections.insert(assignment[i]);
        }
        else
            matchedPairs.push_back(cv::Point(i, assignment[i]));
    }

    /*
    3.3. updating trackers
    update matched trackers with assigned detections.
    each prediction is corresponding to a tracker
    */
    int detIdx, trkIdx;
    for (unsigned int i = 0; i < matchedPairs.size(); i++)
    {
        trkIdx = matchedPairs[i].x;
        detIdx = matchedPairs[i].y;
        trackers[trkIdx].kalmT.update(detFrameData[fi][detIdx].box);
        trackers[trkIdx].score=(detFrameData[fi][detIdx].score);
        trackers[trkIdx].label=(detFrameData[fi][detIdx].label);
    }

    // create and initialize new trackers for unmatched detections
    for (auto umd : unmatchedDetections)
    {
        KalmanTracker tracker = KalmanTracker(detFrameData[fi][umd].box,sort_kal_index);
        Kal_Label_Struct kls={
            tracker,detFrameData[fi][umd].score,detFrameData[fi][umd].label
        };

        trackers.push_back(kls);
    }

    // get trackers' output
    for (auto it = trackers.begin(); it != trackers.end();)
    {
        if (((*it).kalmT.m_time_since_update < 1) &&
            ((*it).kalmT.m_hit_streak >= min_hits || fi <= min_hits))
        {
            SORT::TrackingBox res;
            res.box = (*it).kalmT.get_state();
            res.id = (*it).kalmT.m_id + 1;
            res.frame = fi;
            res.label = (*it).label;
            res.score = (*it).score;
            frameTrackingResult.push_back(res);
            it++;
        }
        else
            it++;

        // 导致第一个没有删除的bug，可能导致突然加一或者减一
        // // remove dead tracklet
        // if (it != trackers.end() && (*it).kalmT.m_time_since_update > max_age){          
        //     it = trackers.erase(it);
        // }
    }

    // remove dead tracklet
    for (auto it = trackers.begin(); it != trackers.end();){
        if ( (*it).kalmT.m_time_since_update > max_age){          
            it = trackers.erase(it);
        }else{
            ++it;
        }
    }

    // std::cout << " ---------. "<< std::endl; 
    // for(auto res:trackers){
    //     std::cout << " kalm_id: " << res.kalmT.m_id ;
    // }
    // std::cout << " --. "<< std::endl; 
    // for(auto res:frameTrackingResult){
    //     std::cout << " box_id: " << res.id ;
    // }
    // std::cout << " --. "<< std::endl; 
    // std::cout << std::endl; 

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    //std::cout << "SORT time : " << duration.count() << " ms" << std::endl;

    return frameTrackingResult;

}

