// 车检测，有车进行车牌检测，有车牌进行车牌识别 （跟踪有点麻烦）

#pragma once

#include "opencv2/core.hpp"
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "post/base.h"
#include "post/model_base.h"

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


class carPlateDetRec {
public:
	std::shared_ptr<detYOLO>  car_model;
	std::shared_ptr<lmYOLO>  plate_model;
	std::shared_ptr<recCarPlate>  crnnNet;	
    cv::Point2f points_ref[4] = { 
		cv::Point2f(0, 0),
		cv::Point2f(100, 0),
		cv::Point2f(100, 32),
		cv::Point2f(0, 32) };

    void init(std::string model_path,int modleClassNum=1,int gpu_id=0)  {
        car_model = std::make_shared<detYOLO>();
        car_model->init("m_enc.pt",1,0); 
        plate_model = std::make_shared<YOLO>();
        plate_model->init("det.rknn",0,2);
        crnnNet = std::make_shared<recCarPlate>();
        crnnNet->init("rec.rknn",0,3);
    }

    std::vector<BoxInfo> infer(cv::Mat img1)  {
        cv::Mat img=img1.clone();
        vector<BoxInfo> boxes= car_model.infer(img1);
        for(const auto& box : boxes){

        string code="";
        BoxInfo carCodeBox;
        cv::Mat carImg = img(cv::Rect(box.x1,box.y1,(box.x2 - box.x1),(box.y2 - box.y1)));
            std::vector<CarBoxInfo>    carPlateBoxes = plateDetPtr_->infer(carImg);
            int index;
            index = chooseBestCarPlateIndex(carPlateBoxes,box.y2 - box.y1);
            if(index != -1){             
            if(carPlateBoxes.size())
            {
                int xmin=int(carPlateBoxes[index].x1);
                int ymin=int(carPlateBoxes[index].y1);
                int xmax=int(carPlateBoxes[index].x2);
                int ymax=int(carPlateBoxes[index].y2);

                cv::Point2f points[] = {	
                    cv::Point2f(float(carPlateBoxes[index].landmark[0]-xmin), float(carPlateBoxes[index].landmark[1]-ymin)),
                    cv::Point2f(float(carPlateBoxes[index].landmark[2]-xmin), float(carPlateBoxes[index].landmark[3]-ymin)),
                    cv::Point2f(float(carPlateBoxes[index].landmark[4]-xmin), float(carPlateBoxes[index].landmark[5]-ymin)),
                    cv::Point2f(float(carPlateBoxes[index].landmark[6]-xmin), float(carPlateBoxes[index].landmark[7]-ymin))
                };
                cv::Mat M = cv::getPerspectiveTransform(points, points_ref);
                cv::Mat img_box = carImg(cv::Rect(xmin, ymin,xmax-xmin,ymax-ymin));

                cv::Mat processed; //掰直后的车牌img
                cv::warpPerspective(img_box,processed, M, cv::Size(100, 32));
                carCodeBox.x1 = carPlateBoxes[index].x1 + box.x1;
                carCodeBox.x2 = carPlateBoxes[index].x2 + box.x1;
                carCodeBox.y1 = carPlateBoxes[index].y1 + box.y1;
                carCodeBox.y2 = carPlateBoxes[index].y2 + box.y1;
                carCodeBox.label = carPlateBoxes[index].label;
                carCodeBox.score = carPlateBoxes[index].score;               
                if(carPlateBoxes[index].label == 1)
                {
                    processed = get_split_merge(processed);
                }
                code = plateRecPtr_->infer(processed);
            }
            }

        }
        return output;
    }

};