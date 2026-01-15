#pragma once

#include "infer/v8/yolo.hpp"
#include "post/base.h"

class detYOLO {
public:
    std::shared_ptr<YOLO> yolo_model;
    void init(std::string model_path,int modleClassNum=1,int npu_id=0)  {
        if(model_path=="smoking_detection/m1_enc.pt" ){
            yolo_model = std::make_shared<YOLO>(model_path,800, 800);
        }else{
            yolo_model = std::make_shared<YOLO>(model_path,1280, 1280);
        }
    }

    std::vector<BoxInfo> infer(cv::Mat img)  {
        std::vector<BoxInfo> output;
        output = yolo_model->detect_img(img);
        return output;
    }

    std::vector<std::vector<BoxInfo>> infer_batch(std::vector<cv::Mat> imgs)  {
        std::vector<std::vector<BoxInfo>> outputs;
        for(auto img:imgs){
            std::vector<BoxInfo> output = yolo_model->detect_img(img);
            outputs.push_back(output);
        }
        return outputs;
    }
};

#ifdef USE_SEG
#include "infer/tensorrt_mask_v8_que/yolo_que.hpp"
// #include "infer/v8_seg/main2_trt_infer.h"


class segYOLO {
public:
    std::shared_ptr<YOLO_QUE> yolo_model;
    void init(std::string model_path,int modleClassNum=1,int npu_id=0)  {
        yolo_model = std::make_shared<YOLO_QUE>(model_path);
    }

    std::vector<Object> infer(cv::Mat& img)  {
        std::vector<Object> output;
        Det_Struct x;
        bool isGet = false;
        cv::Mat img_copy = img.clone();
        x = yolo_model->detect_img(img_copy,isGet);
        if(!isGet)
            return output;
        output=x.objects;
        img=x.img;
        return output;
    }

    // std::vector<Object> infer(cv::Mat& img)  {
    //     std::vector<Object> output;        
    //    output = yolo_model->detect_img(img);        
    //     return output;
    // }

    std::vector<std::vector<BoxInfo>> infer_batch(std::vector<cv::Mat> imgs)  {
        // std::vector<Object> output;
        std::vector<std::vector<BoxInfo>> outputs;
        // outputs = yolo_model->detect_v8(imgs);
        return outputs;
    }
};

#endif


// 人脸 trt
#ifdef USE_FACE
#include "infer/face/face_det.h"
class faceYOLO {
public:
    std::shared_ptr<YOLOV7_face> yolo_model;
    void init(std::string model_path, int modleClassNum=1,int npu_id=0)  {
        Net_config YOLOV7_face_cfg = { 0.3, 0.4, model_path }; //conf=0.1
        yolo_model = std::make_shared<YOLOV7_face>(YOLOV7_face_cfg);
    }

    std::vector<Yolov5Face_BoxStruct> infer(cv::Mat  img,float conf_threshold=0.1)  {
        vector<Yolov5Face_BoxStruct>  output=yolo_model->detect(img);   
        return output;
    }

};

#include "infer/face/face_rec.h"
class faceRecognizer {
public:
    std::shared_ptr<FaceRecognizer> recognizer;
    void init(std::string model_path, int modleClassNum=1,int npu_id=0)  {
        recognizer = std::make_shared<FaceRecognizer>(model_path);
    }

    vector<float> infer(cv::Mat  aligned_face1)  {
        vector<float>  feature1 = recognizer->feature(aligned_face1);         
        return feature1;
    }
};

#endif