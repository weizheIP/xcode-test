#ifndef TORCH_H
#define TORCH_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/types_c.h>

#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <torch/script.h>
#include <torch/torch.h> 
#include <algorithm>
#include <iostream>
#include <time.h>
#include <opencv2/dnn.hpp>

#include "time.h"
#include "sys/time.h"

#include <stdio.h>

#include <strstream>
#include "post/queue.h"
#include "post/base.h"

#include "license.h"
#include <unistd.h>

// struct Detection
// {
//     cv::Rect bbox;
//     float score;
//     int class_idx;
// };

// typedef struct BoxInfo
// {
// 	float x1;
// 	float y1;
// 	float x2;
// 	float y2;
// 	float score;
// 	int label;
// } BoxInfo;


enum Det
{
    tl_x = 0,
    tl_y = 1,
    br_x = 2,
    br_y = 3,
    score = 4,
    class_idx = 5
};

// struct AsynData
// {
//     std::vector<cv::Mat> srcImgs;
//     std::vector<cv::Mat> showImgs;
//     std::vector<std::vector<float>> padInfos;
//     torch::Tensor preds;
//     std::vector<std::string> beLongRtsps;
//     std::vector<std::vector<BoxInfo>> boxeses;

//     // std::vector<torch::Tensor> imgTensors;
// };



class Torch
{
public:


    torch::jit::script::Module module;

    void ScaleCoordinates(std::vector<BoxInfo> &data, float pad_w, float pad_h,
                          float scale, const cv::Size &img_shape)
    {
        auto clip = [](float n, float lower, float upper)
        {
            return std::max(lower, std::min(n, upper));
        };

        // std::vector<BoxInfo> detections;
        for (auto &i : data)
        {
            float x1 = (i.x1 - pad_w) / scale; // x padding
            float y1 = (i.y1 - pad_h) / scale; // y padding
            float x2 = (i.x2 - pad_w) / scale; // x padding
            float y2 = (i.y2 - pad_h) / scale; // y padding

            i.x1 = clip(x1, 0, img_shape.width);
            i.y1 = clip(y1, 0, img_shape.height);
            i.x2 = clip(x2, 0, img_shape.width);
            i.y2 = clip(y2, 0, img_shape.height);

            // i.bbox = cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2));
        }
    }

    torch::Tensor xywh2xyxy(const torch::Tensor &x)
    {
        auto y = torch::zeros_like(x);
        // convert bounding box format from (center x, center y, width, height) to (x1, y1, x2, y2)
        y.select(1, Det::tl_x) = x.select(1, 0) - x.select(1, 2).div(2);
        y.select(1, Det::tl_y) = x.select(1, 1) - x.select(1, 3).div(2);
        y.select(1, Det::br_x) = x.select(1, 0) + x.select(1, 2).div(2);
        y.select(1, Det::br_y) = x.select(1, 1) + x.select(1, 3).div(2);
        return y;
    }

    void Tensor2Detection(const at::TensorAccessor<float, 2> &offset_boxes,
                          const at::TensorAccessor<float, 2> &det,
                          std::vector<cv::Rect> &offset_box_vec,
                          std::vector<float> &score_vec)
    {

        for (int i = 0; i < offset_boxes.size(0); i++)
        {
            offset_box_vec.emplace_back(
                cv::Rect(cv::Point(offset_boxes[i][Det::tl_x], offset_boxes[i][Det::tl_y]),
                         cv::Point(offset_boxes[i][Det::br_x], offset_boxes[i][Det::br_y])));
            score_vec.emplace_back(det[i][Det::score]);
        }
    }

    std::vector<std::vector<BoxInfo>> PostProcessing_v8(const torch::Tensor &detections,
                                                    std::vector<std::vector<float>>pad_info, const cv::Size &img_shape,
                                                    float conf_thres, float iou_thres)
    {
        
        constexpr int item_attr_size = 4;
        
        int batch_size = detections.size(0);
        // number of classes, e.g. 80 for coco dataset
        auto num_classes = detections.size(2) - item_attr_size;

        // get candidates which object confidence > threshold
        

        std::vector<std::vector<BoxInfo>> output(batch_size,std::vector<BoxInfo>());
        // output.reserve(batch_size);

        // iterating all images in the batch
        for (int batch_i = 0; batch_i < batch_size; batch_i++)
        {
            // apply constrains to get filtered detections for current image
            auto det = detections[batch_i].view({-1, num_classes + item_attr_size});

            // if none detections remain then skip and start to process next image
            if (0 == det.size(0))
            {
                // std::vector<BoxInfo> det_vec;
                // output[batch_i]=det_vec;
                continue;
            }
            
            // compute overall score = obj_conf * cls_conf, similar to x[:, 5:] *= x[:, 4:5]
            

            // box (center x, center y, width, height) to (x1, y1, x2, y2)
            torch::Tensor box = xywh2xyxy(det.slice(1, 0, 4));

            // [best class only] get the max classes score at each result (e.g. elements 5-84)
            std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(det.slice(1, item_attr_size, item_attr_size + num_classes), 1);

            // class score
            auto max_conf_score = std::get<0>(max_classes);
            // index
            auto max_conf_index = std::get<1>(max_classes);

            // 插入维度
            max_conf_score = max_conf_score.to(torch::kFloat).unsqueeze(1);
            max_conf_index = max_conf_index.to(torch::kFloat).unsqueeze(1);

            // shape: n * 6, top-left x/y (0,1), bottom-right x/y (2,3), score(4), class index(5)
            det = torch::cat({box.slice(1, 0, 4), max_conf_score, max_conf_index}, 1);

            // for batched NMS
        //     constexpr int max_wh = 4096;
        // auto c = det.slice(1, item_attr_size, item_attr_size + 1) * max_wh;
        // auto offset_box = det.slice(1, 0, 4) + c;
            auto offset_box = det.slice(1, 0, 4);
            

            std::vector<cv::Rect> offset_box_vec;
            std::vector<float> score_vec;

            // copy data back to cpu
            auto offset_boxes_cpu = offset_box.cpu();
            auto det_cpu = det.cpu();
            const auto &det_cpu_array = det_cpu.accessor<float, 2>();

            // use accessor to access tensor elements efficiently
            Tensor2Detection(offset_boxes_cpu.accessor<float, 2>(), det_cpu_array, offset_box_vec, score_vec);

            // run NMS
            std::vector<int> nms_indices;
            cv::dnn::NMSBoxes(offset_box_vec, score_vec, conf_thres, iou_thres, nms_indices);

            std::vector<BoxInfo> det_vec;
            for (int index : nms_indices)
            {
                BoxInfo t;
                const auto &b = det_cpu_array[index];
                // t.bbox =
                //     cv::Rect(cv::Point(b[Det::tl_x], b[Det::tl_y]),
                //              cv::Point(b[Det::br_x], b[Det::br_y]));
                // t.score = det_cpu_array[index][Det::score];
                // t.class_idx = det_cpu_array[index][Det::class_idx];
                t.x1 = b[Det::tl_x];
                t.y1 = b[Det::tl_y];
                t.x2 = b[Det::br_x];
                t.y2 = b[Det::br_y];
                t.score = det_cpu_array[index][Det::score];
                t.label = det_cpu_array[index][Det::class_idx];
                if(t.score==1) //gpu负载大的时候会出很多框
                    continue;
                det_vec.emplace_back(t);
            }

            ScaleCoordinates(det_vec, pad_info[batch_i][0], pad_info[batch_i][1], pad_info[batch_i][2], img_shape);

            // save final detection for the current image
            output[batch_i]=det_vec;
        } // end of batch iterating

        return output;
    }


    std::vector<float> LetterboxImage(const cv::Mat &src, cv::Mat &dst, const cv::Size &out_size)
    {
        timeval tv1,tv1_1,tv1_2,tv1_3,tv1_4,tv2,tv3,tv4,tv5;
        auto in_h = static_cast<float>(src.rows);
        auto in_w = static_cast<float>(src.cols);
        float out_h = out_size.height;
        float out_w = out_size.width;

        float scale = std::min(out_w / in_w, out_h / in_h);
        
        bool widthBigger = (out_w / in_w) < (out_h / in_h) ? true:false;

        int mid_h = static_cast<int>(in_h * scale);
        int mid_w = static_cast<int>(in_w * scale);
        gettimeofday(&tv1_1, NULL);
        if(widthBigger)
        {
            if(in_w != mid_w)
                cv::resize(src, dst, cv::Size(mid_w, mid_h));
            else
                dst = src;
        }
        else
        {
            if(in_h != mid_h)
                cv::resize(src, dst, cv::Size(mid_w, mid_h));
            else
                dst = src;
        }

        

gettimeofday(&tv1_2, NULL);
        int top = (static_cast<int>(out_h) - mid_h) / 2;
        int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
        int left = (static_cast<int>(out_w) - mid_w) / 2;
        int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

        cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
gettimeofday(&tv1_3, NULL);
        std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};

        // std::cout<<"LetterboxImage 1 Time:"<<(tv1_2.tv_sec - tv1_1.tv_sec)*1000+(tv1_2.tv_usec-tv1_1.tv_usec)/1000
        //     <<",2 Time"<<(tv1_3.tv_sec - tv1_2.tv_sec)*1000+(tv1_3.tv_usec-tv1_2.tv_usec)/1000
        //     <<",3 Time"<<(tv1_4.tv_sec - tv1_3.tv_sec)*1000+(tv1_4.tv_usec-tv1_3.tv_usec)/1000
        //     <<std::endl;
        return pad_info;
    }

    Torch(int model_w_,int model_h_):model_w(model_w_),model_h(model_h_)   
    {;}
    ~Torch()
    {
        // #ifdef _SelCard
        if(_device)
            delete _device;
        // #endif
        status_ = false;       
    }

    // #ifdef _SelCard
    void Init(std::string model_path,int gpu_id=0)
    // #else
    // void Init(std::string model_path,int isAsyn = 1)
    // #endif
    {
        modelPath_ = model_path;
        // #ifdef _SelCard
        _gpuId = gpu_id;
        // #endif
        // "/media/ps/data1/liuym/202012-count/yolov5-6.1/runs/train/pig/0308s5/weights/best.torchscript"
        std::cout<<model_path<<std::endl;
        unsigned char *outdata;
        FILE *f = fopen(model_path.c_str(),"r");
        fseek(f,0,SEEK_END);
        size_t size = ftell(f);
        fseek(f,0,SEEK_SET);
        unsigned char* buff = (unsigned char*)malloc(size);
        fread(buff,size,1,f);
        fclose(f);

        // for(int i = 0;i<20;i++){
        //     std::cout<<(int)(buff[i])<<std::endl;
        // }
        
        get_model_decrypt_value(buff,outdata,size);
        std::strstreambuf sbuf(buff, size);
        std::istream ins(&sbuf);

        // module = torch::jit::load(model_path.c_str(),torch::kCUDA);
        // torch::jit::getProfilingMode() = false;
        // torch::jit::getExecutorMode() = false;
        torch::jit::setGraphExecutorOptimize(false);

        // #ifdef _SelCard
        // torch::Device device(torch::kCUDA, gpu_id);
        _device = new torch::Device(torch::kCUDA, gpu_id);
        module = torch::jit::load(ins,torch::Device(torch::DeviceType::CUDA,gpu_id));

        vector<int> input_sizes = {1280,800,640};
        for(int i = 0; i < input_sizes.size(); i++) {
            auto test_input = torch::randn({1, 3, input_sizes[i], input_sizes[i]}).to(*_device);
            try {
                // 运行模型
                auto output = module.forward({test_input}).toTensor();
                std::cout << "输入 " << input_sizes[i] << "x" << input_sizes[i] << " ，输出尺寸: " << output.sizes() << std::endl;
                model_w=model_h=input_sizes[i];
                break;
            } catch (const std::exception& e) {
                // std::cout << "输入 " << input_sizes[i] << "x" << input_sizes[i] << " 无效: " << e.what() << std::endl;
            }
        }

        // auto graph = module.get_method("forward").graph();
        // for (auto* input_value : graph->inputs()) {
        //     // 跳过 self
        //     if (input_value->type()->isSubtypeOf(torch::jit::TensorType::get())) {
        //         auto tensor_type = input_value->type()->expect<torch::jit::TensorType>();
        //         if (tensor_type && tensor_type->sizes().isComplete()) {
        //             auto symbolic_shape = tensor_type->sizes();
        //             auto sizes = symbolic_shape.sizes().value();
        //             int batch_size = sizes[0].has_value() ? sizes[0].value() : -1;
        //             int channels   = sizes[1].has_value() ? sizes[1].value() : -1;
        //             int height     = sizes[2].has_value() ? sizes[2].value() : -1;
        //             int width      = sizes[3].has_value() ? sizes[3].value() : -1;
        //             std::cout << "Model expects input shape: ["
        //                     << batch_size << ", "
        //                     << channels << ", "
        //                     << height << ", "
        //                     << width << "]" << std::endl;
        //             break; // 找到第一个张量输入就可以退出
        //         }
        //     }
        // }



    //     auto graph = module.graph();
    // auto graph = module.get_method("forward").graph();
    
    // // 遍历输入节点（通常第一个输入为数据输入）
    // for (auto input : graph->inputs()) {
    //     // 过滤掉权重等参数，只关注输入张量（通常名称为 "input" 或类似）
    //     if (input->type()->isSubtypeOf(torch::jit::TensorType::get())) {
    //         auto tensor_type = input->type()->cast<torch::jit::TensorType>();
            
    //         // 获取输入张量的尺寸（可能包含动态维度，用 -1 表示）
    //         auto sizes = tensor_type->sizes();
            
    //         std::cout << "输入尺寸: ";
    //         for (auto s : sizes) {
    //             std::cout << s << " ";
    //         }
    //         std::cout << std::endl;
            
    //         // 若尺寸包含动态维度（-1），可能需要结合模型文档确定默认值
    //         // 例如：常见的图像输入格式为 [batch, channel, height, width]
    //         if (sizes.size() == 4) {
    //             int height = sizes[2];
    //             int width = sizes[3];
    //             std::cout << "推断宽高: height=" << height << ", width=" << width << std::endl;
    //         }
    //     }
    // }


        // module.to(*_device);
        // #else
        // module = torch::jit::load(ins,torch::kCUDA);
        // #endif
        free(buff);
        
        status_ = true;       
           
    }
    
    
    std::vector<std::vector<BoxInfo>> detect_v8(std::vector<cv::Mat> imgs)
    {
        // cv::Mat img = cv::imread("/media/ps/data1/liuym/data/1.jpg");

        // std::cout<<"Begin Detec,Path"<<vpath<<std::endl;
        timeval tv1,tv1_1,tv1_2,tv1_3,tv1_4,tv2,tv3,tv4,tv5;
        gettimeofday(&tv1, NULL);

        clock_t start = clock();
        // Preparing input tensor
        std::vector<torch::Tensor> imgTensors;
        std::vector<std::vector<float>> pad_info;
        for(int size_i = 0;size_i<imgs.size();size_i++){
            
            cv::Mat img = imgs[size_i].clone();
            
            pad_info.push_back(LetterboxImage(img, img, cv::Size(model_w, model_h)));
            gettimeofday(&tv1_1, NULL);
            cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
            gettimeofday(&tv1_2, NULL);
            // #ifdef _SelCard
            // torch::TensorOptions options = torch::TensorOptions().device(*_device);
            torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3},torch::kByte).cuda().to(*_device);//
            // #else
            // torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3},torch::kByte).cuda();//
            // #endif
            gettimeofday(&tv1_3, NULL);
            imgTensor = imgTensor.permute({2, 0, 1});
            imgTensor = imgTensor.toType(torch::kFloat);
            imgTensor = imgTensor.div(255);
            imgTensor = imgTensor.unsqueeze(0);
            imgTensors.push_back(imgTensor);
            gettimeofday(&tv1_4, NULL);
            // std::cout<<"cvt Time:"<<(tv1_2.tv_sec - tv1_1.tv_sec)*1000+(tv1_2.tv_usec-tv1_1.tv_usec)/1000
            // <<",2 Time"<<(tv1_3.tv_sec - tv1_2.tv_sec)*1000+(tv1_3.tv_usec-tv1_2.tv_usec)/1000
            // <<",3 Time"<<(tv1_4.tv_sec - tv1_3.tv_sec)*1000+(tv1_4.tv_usec-tv1_3.tv_usec)/1000
            // <<std::endl;
        }
        
        gettimeofday(&tv2, NULL);
        torch::Tensor imgTensor= torch::cat(imgTensors,0);
        gettimeofday(&tv3, NULL);

        // #ifdef _SelCard
        torch::Tensor preds = module.forward({imgTensor}).toTensor().to(*_device);
        // #else
        // torch::Tensor preds = module.forward({imgTensor}).toTensor().cuda();
        // #endif

        // #ifdef _SelCard
        // // preds = preds.to(*_device);
        // #endif
        
        gettimeofday(&tv4, NULL);
        
        
        auto detections = PostProcessing_v8(preds.permute({0, 2, 1}), pad_info, imgs[0].size(), 0.1f, 0.6f);
        
        // if(detections[0].size()>1){
            // std::cout<<"------------infer error-----------cuda error--------"<<std::endl;
        //     std::vector<std::vector<BoxInfo>> output(1,std::vector<BoxInfo>());
        //     return output; //临时解决大象很多框的bug
        // }

        // std::vector<std::vector<BoxInfo>> detections = PostProcessing(preds, pad_info, imgs[0].size(), 0.01, 0.6);//conf nms


    
        gettimeofday(&tv5, NULL);
        // std::cout<<"LetterboxImage Time:"<<(tv2.tv_sec - tv1.tv_sec)*1000+(tv2.tv_usec-tv1.tv_usec)/1000
        // <<",torch::cat:"<<(tv3.tv_sec - tv2.tv_sec)*1000+(tv3.tv_usec-tv2.tv_usec)/1000
        // <<",module.forward:"<<(tv4.tv_sec - tv3.tv_sec)*1000+(tv4.tv_usec-tv3.tv_usec)/1000
        // <<",PostProcessing:"<<(tv5.tv_sec - tv4.tv_sec)*1000+(tv5.tv_usec-tv4.tv_usec)/1000
        // <<",ALL"<<(tv5.tv_sec - tv1.tv_sec)*1000+(tv5.tv_usec-tv1.tv_usec)/1000
        // <<std::endl;

        // std::cout<<"Forward Time:"<<(float)(t2.tv_sec - t1.tv_sec)+(float)(t2.tv_usec-t1.tv_usec)/1000000.0f
        // <<",All Time:"<<(float)(ta2.tv_sec - ta1.tv_sec)+(float)(ta2.tv_usec-ta1.tv_usec)/1000000.0f
        // <<std::endl;
        // cv::imwrite("out.jpg", out_img);
        return detections;
    }


private:
    bool status_;
    std::string modelPath_;
    int model_w,model_h;
    int _gpuId;
    torch::Device *_device;   

};

#endif

