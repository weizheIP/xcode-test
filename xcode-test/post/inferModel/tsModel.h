#pragma once

#include "infer/torch/torch.h"
#include "post/base.h"

class detYOLO {
public:
    std::shared_ptr<Torch> yolo_model;
    void init(std::string model_path,int modleClassNum=1,int npu_id=0)  {
        // if(model_path=="smoking_detection/m1_enc.pt" ){
        //     yolo_model = std::make_shared<Torch>(800, 800);
        //     yolo_model->Init(model_path,npu_id);
        // }else if( model_path=="m_enc_pig.pt"){
        //     yolo_model = std::make_shared<Torch>(640, 640);
        //     yolo_model->Init(model_path,npu_id);
        // }
        // else{
            yolo_model = std::make_shared<Torch>(1280, 1280);
            yolo_model->Init(model_path,npu_id);
        // }

    }

    std::vector<BoxInfo> infer(cv::Mat img)  {
        std::vector<BoxInfo> output;
        std::vector<cv::Mat> imgs;
        imgs.push_back(img);
        output = yolo_model->detect_v8(imgs)[0];
        return output;
    }

    std::vector<std::vector<BoxInfo>> infer_batch(std::vector<cv::Mat> imgs)  {
        std::vector<std::vector<BoxInfo>> outputs;
        outputs = yolo_model->detect_v8(imgs);
        return outputs;
    }
};

// onnxruntime 车牌检测和车牌识别
#ifdef USE_LANDMARK
#include "infer/car_plate/face_det.h"
class lmYOLO {
public:
    std::shared_ptr<YOLOV7_face> yolo_model;
    void init(std::string model_path, int modleClassNum=1,int npu_id=0)  {
        Net_config YOLOV7_face_cfg = { 0.5, 0.5, model_path };
        yolo_model = std::make_shared<YOLOV7_face>(YOLOV7_face_cfg);
    }

    // std::vector<Yolov5Face_BoxStruct> infer(cv::Mat  img)  {
    //     std::vector<Yolov5Face_BoxStruct> output=    yolo_model->detect(img);  
    //     return output;
    // }

    std::vector<CarBoxInfo> infer(cv::Mat  img,string flag="")  {
        std::vector<Yolov5Face_BoxStruct> output;  
        if(flag=="v8") {
            output=    yolo_model->detect_v8(img);
        }
        else {
            output=    yolo_model->detect(img);  
        }  
        std::vector<CarBoxInfo> output1;
        for(auto& box : output) {
            CarBoxInfo carBox;
            carBox.x1 = box.x1;
            carBox.y1 = box.y1;
            carBox.x2 = box.x2;
            carBox.y2 = box.y2;
            carBox.label = 0;
            carBox.score = box.score;
            carBox.landmark = {box.keypoint[0].x, box.keypoint[0].y,
                               box.keypoint[1].x, box.keypoint[1].y,
                               box.keypoint[2].x, box.keypoint[2].y,
                               box.keypoint[3].x, box.keypoint[3].y};
            output1.push_back(carBox);
        }
        return output1;
    }

};

#include "infer/car_plate/CrnnNet.h"
class recCarPlate {
public:
    std::shared_ptr<CrnnNet> yolo_model;
    void init(std::string model_path, int modleClassNum=1,int npu_id=0)  {
        yolo_model = std::make_shared<CrnnNet>();
        yolo_model->initModel(model_path,"");
    }

    std::string infer(cv::Mat  img)  {
        std::string output= yolo_model->getTextLine(img).text;            
        return output;
    }
};
#endif

// 人脸 onnxruntime
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
        // vector<Yolov5Face_BoxStruct>  output=yolo_model->detect(img);   
        vector<Yolov5Face_BoxStruct>  output=yolo_model->detect_v8(img);   

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

#include "infer/face/face_expression.h"
class faceExpression {
public:
    std::shared_ptr<FaceExpression> recognizer;
    void init(std::string model_path, int modleClassNum=1,int npu_id=0)  {
        recognizer = std::make_shared<FaceExpression>(model_path);
    }

    int infer(cv::Mat  aligned_face1)  {
        int feature1 = recognizer->feature(aligned_face1);         
        return feature1;
    }
};

#endif
