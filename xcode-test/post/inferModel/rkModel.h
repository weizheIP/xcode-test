#pragma once

#include "infer/rknn_Mclass/include/infer.h"
#include "post/base.h"

class detYOLO {
private:
    int modleClassNum=1;
public:
    std::shared_ptr<RKNN_Detector> yolo_model;

    void init(std::string model_path,int modleClassNum=1,int npu_id=0)  {
        yolo_model = std::make_shared<RKNN_Detector>(model_path.c_str());
        if (yolo_model->init(npu_id) == -1)
        {
            std::cout <<  " Algs_APP init fail !" << "\n";
        }
        this->modleClassNum=modleClassNum;
    }

    std::vector<BoxInfo> infer(cv::Mat img)  {
        std::vector<BoxInfo> output;
        int ret = yolo_model->detect(img, output,modleClassNum); //不用传类别，自动获取
        return output;
    }

    std::vector<std::vector<BoxInfo>> infer_batch(std::vector<cv::Mat> imgs)  {        
        std::vector<std::vector<BoxInfo>> outputs;
        for(auto img:imgs){
            std::vector<BoxInfo> output;
            int ret = yolo_model->detect(img, output,modleClassNum);
            outputs.push_back(output);
        }
        return outputs;
    }
};

#ifdef USE_SEG
#include "infer/rk_seg/rk_seg.h"

class segYOLO {
public:
    rk_seg yolo_model;
    void init(std::string model_path,int modleClassNum=1,int npu_id=0)  {
        if (yolo_model.init(model_path) != 0)
        {
            std::cout<<"seg init fail!"<<"\n";
        }
    }

    std::vector<Object> infer(cv::Mat img)  {
        std::vector<Object> output;
        int ret  = yolo_model.seg(img,output);
        return output;
    }
};
#endif


#ifdef USE_LANDMARK
#include "infer/car_plate/car_infer.h"
class lmYOLO {
public:
    std::shared_ptr<CarPlateDet> yolo_model;
    // Torch yolo_model;
    void init(std::string model_path, int modleClassNum=1,int npu_id=0)  {
        yolo_model = std::make_shared<CarPlateDet>((char*)model_path.c_str(),modleClassNum);
        yolo_model->init(npu_id);
    }

    std::vector<CarBoxInfo> infer(cv::Mat  img)  {
        std::vector<CarBoxInfo> output;
        yolo_model->detect(img,output);            
        return output;
    }

    // std::vector<std::vector<BoxInfo>> infer_batch(std::vector<cv::Mat > imgs)  {
    //     std::vector<std::vector<BoxInfo>> outputs;
    //     return outputs;
    // }
};


#include "infer/car_plate/car_infer.h"
class recCarPlate {
public:
    std::shared_ptr<CarPlateRec> yolo_model;
    void init(std::string model_path, int modleClassNum=1,int npu_id=0)  {
        yolo_model = std::make_shared<CarPlateRec>((char*)model_path.c_str());
        yolo_model->init(npu_id);
    }

    std::string infer(cv::Mat  img)  {
        std::string output= yolo_model->get_car_plate_code(img);            
        return output;
    }
};
#endif

// 人脸
#ifdef USE_FACE
#include "infer/face/yolov5face/face2yolov5.h"
class faceYOLO {
public:
    std::shared_ptr<Face_Detector2> yolo_model;
    void init(std::string model_path, int modleClassNum=1,int npu_id=0)  {
        yolo_model = std::make_shared<Face_Detector2>((char*)model_path.c_str());
        yolo_model->init();
    }

    std::vector<Yolov5Face_BoxStruct> infer(cv::Mat  img,float conf_threshold=0.1)  {
        vector<Yolov5Face_BoxStruct> face_result;
        // yolo_model->detect_face(img,face_result,0.4,conf_threshold);   
        yolo_model->detect_face(img,face_result,0.1,0.3);   
        return face_result;
    }

};

#include "infer/face/face_recognition.h"
class faceRecognizer {
public:
    std::shared_ptr<FaceRecognizer> recognizer;
    void init(std::string model_path, int modleClassNum=1,int npu_id=0)  {
        recognizer = std::make_shared<FaceRecognizer>((char*)model_path.c_str());
        recognizer->init();
    }

    vector<float> infer(cv::Mat  aligned_face1)  {
        vector<float>  feature1 = recognizer->feature(aligned_face1);         
        return feature1;
    }
};

#endif

