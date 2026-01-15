#pragma once
#include "post/base.h"
#include "post/model_base.h"
#include "post/post_util.h"
#include "json.h"

class parallelYOLO {
public:

    detYOLO yolo_model;
    detYOLO yolo_model2;
    // detYOLO yolo_model3;
    void init(std::string model_path,int modleClassNum=1,int npu_id=0)  {     
        yolo_model.init(model_path+"/m1_enc.pt",1,npu_id);  //垃圾
        yolo_model2.init(model_path+"/m2_enc.pt",1,npu_id); //人
    }

    std::vector<BoxInfo> infer(cv::Mat img)  {
        std::vector<BoxInfo> boxes_out,boxes2,boxes_tmp;     
        boxes_out = yolo_model.infer(img.clone());
        boxes2 = yolo_model2.infer(img.clone());
        for(auto box:boxes2)        {
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


extern Json::Value g_value;
extern std::mutex g_lock_json;

class chargeGunYOLO {
public:

    detYOLO yolo_model;
    detYOLO yolo_model2;
    detYOLO yolo_model3;
    void init(std::string model_path,int modleClassNum=1)  {     
        yolo_model.init(model_path+"/m_enc.pt",1);  //chargeGun-2
        yolo_model2.init(model_path+"/m2_enc.pt",1); //车-1
        yolo_model3.init(model_path+"/m3_enc.pt",1); //人-2
    }

    std::vector<BoxInfo> infer(cv::Mat img,std::string rtsp_path)  {

        g_lock_json.lock();
        Json::Value G_config = Get_Rtsp_Json1(g_value,rtsp_path,"cv_charging_gun_notputback")["config"];
        Json::Value Roi = Get_Rtsp_Json1(g_value,rtsp_path,"cv_charging_gun_notputback")["roi"];
        g_lock_json.unlock();

        float conf = G_config["score"].asFloat();

        vector<BoxInfo>  detectResults,data_result,result2,result3,result4 ;
        vector<BoxInfo>  detectResults2 ;


        std::vector<std::vector<cv::Point>> contourss; //检测区域
        std::vector<std::vector<cv::Point>> contourss2;//充电桩区域
    
        Json::Value roi2 = G_config["charging_pile_roi"];

        for (int i = 0; i < Roi.size(); i++)//检测区域
        {
            std::vector<cv::Point> contours;
            for (int j = 0; j < Roi[i].size(); j++)
                contours.push_back(cv::Point(Roi[i][j][0].asInt(), Roi[i][j][1].asInt()));//
            contourss.push_back(contours);
        }

        for (int i = 0; i < roi2.size(); i++)//充电桩区域
        {
            std::vector<cv::Point> contours;
            for (int j = 0; j < roi2[i].size(); j++)
                contours.push_back(cv::Point(roi2[i][j][0].asInt(), roi2[i][j][1].asInt()));//
            contourss2.push_back(contours);
        }

        bool NeedCheckPeoAndCar = false;
        for(int i = 0; i < roi2.size(); i++)
        {
            bool gunOnTree = false;
            //充电桩检测，整体，把泊位也算进来
            // #ifdef __DETLITTLEPIC__
            int x,y,w,h;
            getGiggerOutSizeReat2(x,y,w,h,roi2[i],Roi[i]);

            // if(G_parms.DeBugShow)
            // {
            //     std::vector<cv::Point> roiss;
            //     roiss.push_back(cv::Point(x,y));
            //     roiss.push_back(cv::Point(x+w,y));
            //     roiss.push_back(cv::Point(x+w,y+h));
            //     roiss.push_back(cv::Point(x,y+h));
            //     cv::polylines(data->rawImage, roiss, true, cv::Scalar(0, 0, 255), 1);
            // }
            cv::Mat ChargeTreeRoi = img(cv::Rect(x,y,w,h));
            std::vector<BoxInfo> result;
            // _yoloPtr->detect(ChargeTreeRoi,result,2);
            result=yolo_model.infer(ChargeTreeRoi.clone());
            //判断充电器有没有在充电桩上，无则识别车辆和行人
            for(auto item : result)
            {
                // if(G_parms.DeBugShow)
                // {
                //     BoxInfo tmp = item;
                //     tmp.x1 = x + item.x1;
                //     tmp.y1 = y + item.y1;
                //     tmp.x2 = x + item.x2;
                //     tmp.y2 = y + item.y2;
                //     tmp.score = roundToTwoDecimalPlaces(item.score);
                //     tmp.label = item.label;
                //     detectResults2.push_back(tmp);
                // }
                
                if(item.label != 0 || item.score < conf)
                {
                    continue;
                }
                cv::Point centerPoint((x + (item.x1 + item.x2)) / 2,(y + (item.y1 + item.y2))/2);
                if(cv::pointPolygonTest(contourss2[i],centerPoint,true) >= 0)
                {
                    // if(G_parms.DeBugShow)
                    //     std::cout<<"[Debug] gun on tree !\n";
                    gunOnTree = true;
                    break;
                }
            }
            //把充电枪目标存放到data->result, 到后面统一过滤
            for(auto item : result)
            {
                if(item.label != 0 || item.score < conf)
                {
                    continue;
                }
                BoxInfo tmp;
                tmp.x1 = item.x1 + x;
                tmp.y1 = item.y1 + y;
                tmp.x2 = item.x2 + x;
                tmp.y2 = item.y2 + y;
                tmp.label = item.label;
                tmp.score = item.score;
                data_result.push_back(tmp);
            }

            if(!gunOnTree && !NeedCheckPeoAndCar)
            {
                NeedCheckPeoAndCar = true;
                // _yoloPtr2->detect(data->rawImage,data->result2,2);
                // _yoloPtr3->detect(data->rawImage,data->result3,2);   
                // _yoloPtr->detect(data->rawImage,data->result4,2);  

                
                result4 = yolo_model.infer(img.clone());
                result2 = yolo_model2.infer(img.clone());
                result3 = yolo_model3.infer(img.clone());
            }
        }



        // 桩/车/人 过滤
        for(int i = 0; i < contourss2.size(); i++)
        {
            bool isOutRoi = false;
            bool isOnChangeTree = false;
            ChargeGunReCheck:
            for(int j = 0; j < data_result.size(); j++)
            {   //枪在检测区域内就删除
                cv::Point CenterPoint((data_result[j].x1 + data_result[j].x2) / 2, (data_result[j].y1 + data_result[j].y2) / 2);
                if(cv::pointPolygonTest(contourss2[i], CenterPoint, true) >= 0)
                {
                    data_result.erase(data_result.begin()+j);
                    isOnChangeTree = true;//对应的充电桩如果有枪，车位的情况就不判断了
                    goto ChargeGunReCheck;
                }

            }
            if(isOnChangeTree)
            {
                //对应的充电桩如果有枪，车位的情况就不判断了,还得把泊位里的充电器删掉
                ChargeGunReCheck2:
                for(int j = 0; j < data_result.size(); j++)
                {   //枪在检测区域内就删除
                    cv::Point CenterPoint((data_result[j].x1 + data_result[j].x2) / 2, (data_result[j].y1 + data_result[j].y2) / 2);
                    if(cv::pointPolygonTest(contourss[i], CenterPoint, true) >= 0)
                    {
                        data_result.erase(data_result.begin()+j);
                        goto ChargeGunReCheck2;
                    } 
                }
                continue;
            }

            
            ChargeGunReCheck3:
            for(int j = 0; j < data_result.size(); j++)
            {
                cv::Point CenterPoint((data_result[j].x1 + data_result[j].x2) / 2, (data_result[j].y1 + data_result[j].y2) / 2);
                if(cv::pointPolygonTest(contourss[i], CenterPoint, true) >= 0)
                {
                    //检测是否有枪在泊位内
                    for(int z = 0; z < result2.size(); z++)
                    {
                        //如果车在里面删除
                        if(isInRoi(result2[z],contourss[i]))
                        {
                            data_result.erase(data_result.begin()+j);
                            // if(G_parms.DeBugShow)
                            //     std::cout<<"[Debug] 车在里面删除 !\n";
                            goto ChargeGunReCheck3;
                        }
                    }
                    //是否与人有交集
                    for(int z = 0; z < result3.size(); z++)
                    {
                        if(GET_IOU2(result3[z].x1,result3[z].y1,(result3[z].x2 - result3[z].x1),(result3[z].y2 - result3[z].y1),
                        data_result[j].x1,data_result[j].y1,(data_result[j].x2 - data_result[j].x1),(data_result[j].y2 - data_result[j].y1)
                        ))
                        {
                            data_result.erase(data_result.begin()+j);
                            // if(G_parms.DeBugShow)
                            //     std::cout<<"[Debug] 与人有交集删除 !\n";
                            goto ChargeGunReCheck3;
                        }
                    }
                    
                }
            }
        }

        //删除在roi外的充电枪
        ChargeGunReCheck4:
        for(int i = 0; i < data_result.size(); i++)
        {
            bool isInRoi = false;
            for(int j = 0; j < contourss.size(); j++)
            {
                cv::Point CenterPoint((data_result[i].x1 + data_result[i].x2) / 2, (data_result[i].y1 + data_result[i].y2) / 2);
                if(cv::pointPolygonTest(contourss[j], CenterPoint, true) >= 0)
                {
                    isInRoi = true;
                    break;
                }
            }
            if(!isInRoi)
            {
                data_result.erase(data_result.begin()+i);
                goto ChargeGunReCheck4;
            }

        }
        //删除切图不全导致检测出来的充电枪原本是插在车上的，但是识别出来是地上的
        ChargeGunReCheck5:
        for(int i = 0; i < data_result.size(); i++)
        {
            bool labelNotRight = false;
            for(int j = 0; j < result4.size(); j++)
            {
                int cx1 = (data_result[i].x1 + data_result[i].x2) / 2;
                int cy1 = (data_result[i].y2 + data_result[i].y1) / 2;
                int cx2 = (result4[j].x1 + result4[j].x2) / 2;
                int cy2 = (result4[j].y2 + result4[j].y1) / 2;
                int drift = std::abs(cx2 - cx1) + std::abs(cy2 - cy1);
                if(drift < 100 && data_result[i].label == 1)
                {
                    labelNotRight = true;
                    std::cout<<"[WARNing] chargeGUn label not right !\n";
                    break;
                }
            }
            if(labelNotRight)
            {
                data_result.erase(data_result.begin()+i);
                goto ChargeGunReCheck5;
            }

        }


        // RoiFillter2(data, contourss, contourss2);
        detectResults = data_result;
        return detectResults;




        // std::vector<BoxInfo> boxes_out,boxes2,boxes3,boxes_tmp;     
        // boxes_out = yolo_model.infer(img.clone());
        // boxes2 = yolo_model2.infer(img.clone());
        // boxes3 = yolo_model3.infer(img.clone());
        // for(auto box:boxes2)        {
        //     box.label+=2; 
        //     boxes_tmp.push_back(box);
        // }     
        // for(auto box:boxes3)        {
        //     box.label+=3; 
        //     boxes_tmp.push_back(box);
        // }   
        // boxes_out.insert(boxes_out.end(), boxes_tmp.begin(), boxes_tmp.end());         
        // return boxes_out;
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
        boxes2 = yolo_model2->infer(img.clone());


        std::vector<cv::Mat> matArr;
        std::vector<std::vector<BoxInfo>> smk_boxes;
        for(auto box:boxes2)
        {
            cv::Mat smkImg;
            if(box.label == 0) //人
            {
                cv::Mat smkImg = img(cv::Rect(box.x1,box.y1,box.x2 - box.x1,box.y2 - box.y1));
                matArr.push_back(smkImg);
            }
            if (algName == "smoking_detection"  || algName == "phone_detection"|| algName == "play_phone_detection"){
                box.label+=1; //吸烟和手机一个类
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
        for(auto img:imgs){
            std::vector<BoxInfo> output = infer(img);
            outputs.push_back(output);
        }
        return outputs;
    }
};

 