#pragma once
#include "post/base.h"
#include "post/model_base.h"

class parallelYOLO {
public:

    detYOLO yolo_model;
    detYOLO yolo_model2;
    std::string algName;
    void init(std::string model_path,int modleClassNum=1,int npu_id=0)  {     
        yolo_model.init(model_path+"/m1_enc.pt",1,npu_id);  //垃圾
        yolo_model2.init(model_path+"/m2_enc.pt",1,npu_id); //人
        algName=model_path;
    }

    std::vector<BoxInfo> infer(cv::Mat img)  {
        std::vector<BoxInfo> boxes_out,boxes2,boxes_tmp;     
        boxes_out = yolo_model.infer(img.clone());
        boxes2 = yolo_model2.infer(img.clone());
        for(auto box:boxes2)        {
            if (algName == "phone_detection"|| algName == "play_phone_detection" )
                box.label+=1; //手机一个类
            else
                box.label+=4; //大件行李和垃圾4个类           
            boxes_tmp.push_back(box);
        }     
        boxes_out.insert(boxes_out.end(), boxes_tmp.begin(), boxes_tmp.end());         
        return boxes_out;
    }

};

class parallelYOLO3 {
public:

    detYOLO yolo_model;
    detYOLO yolo_model2;
    detYOLO yolo_model3;
    // 异物入侵：人2车1动物3
    void init(std::string model_path,int modleClassNum=1,int npu_id=0)  {     
        yolo_model.init(model_path+"/m_enc.pt",1,npu_id);  //人
        yolo_model2.init(model_path+"/m1enc.pt",1,npu_id); //车
        yolo_model3.init(model_path+"/m2enc.pt",1,npu_id); //动物
    }

    std::vector<BoxInfo> infer(cv::Mat img)  {
        std::vector<BoxInfo> boxes_out,boxes2,boxes3,boxes_tmp;     
        boxes_out = yolo_model.infer(img.clone());
        boxes2 = yolo_model2.infer(img.clone());
        boxes3 = yolo_model3.infer(img.clone());
        for(auto box:boxes2)        {
            box.label+=2;     
            boxes_tmp.push_back(box);
        }   
        for(auto box:boxes3)        {
            box.label+=3;         
            boxes_tmp.push_back(box);
        }     
        boxes_out.insert(boxes_out.end(), boxes_tmp.begin(), boxes_tmp.end());         
        return boxes_out;
    }

};

class serialYOLO {
public:
    std::shared_ptr<detYOLO> yolo_model;
    std::shared_ptr<detYOLO> yolo_model2;
    std::string algName;
    void init(std::string model_path,int modleClassNum=1,int npu_id=0)  {
        yolo_model = std::make_shared<detYOLO>();//(800, 800);
        yolo_model2 = std::make_shared<detYOLO>();//(1280, 1280);
        yolo_model->init(model_path+"/m1_enc.pt",1, npu_id);
        yolo_model2->init(model_path+"/m2_enc.pt",2, npu_id); //人
        algName=model_path;
    }

    std::vector<BoxInfo> infer(cv::Mat img)  {
        std::vector<BoxInfo> boxes_out,boxes2,boxes_tmp;        
        // std::vector<cv::Mat> tempV2;
        // tempV2.emplace_back(img);
        boxes2 = yolo_model2->infer(img);

        std::vector<cv::Mat> matArr;
        std::vector<std::vector<BoxInfo>> smk_boxes; 
        for(auto box:boxes2) //教室人多要70*50=4s
        {
            cv::Mat smkImg;
            if(box.label == 0) //人
            {
                cv::Mat smkImg = img(cv::Rect(box.x1,box.y1,box.x2 - box.x1,box.y2 - box.y1));
                matArr.push_back(smkImg);
            }
            if (algName == "smoking_detection" ){
                box.label+=1; //吸烟
            } 
            if (algName == "charging_pile_occupied_by_oil_car" ){
                box.label+=2; //绿蓝
            }                
            
            boxes_tmp.push_back(box);
        }              
        if(matArr.size() != 0)
            smk_boxes = yolo_model->infer_batch(matArr);         
        int smokeIndex = 0;
        for(int i = 0; i < boxes2.size();i++)
        {
            if(boxes2[i].label == 1)  //人
                continue;
            for(auto smk_box:smk_boxes[smokeIndex])
            {
                smk_box.x1 += boxes2[i].x1;
                smk_box.y1 += boxes2[i].y1;
                smk_box.x2 += boxes2[i].x1;
                smk_box.y2 += boxes2[i].y1;
                boxes_out.push_back(smk_box);
                
            }
            smokeIndex++;
        }
        boxes_out.insert(boxes_out.end(), boxes_tmp.begin(), boxes_tmp.end());         
        return boxes_out;
    }

    std::vector<std::vector<BoxInfo>> infer_batch(std::vector<cv::Mat> imgs)  {
        std::vector<std::vector<BoxInfo>> outputs;
        for(const auto &img:imgs){
            std::vector<BoxInfo> output = infer(img);
            outputs.push_back(output);
        }
        return outputs;
    }
};

 