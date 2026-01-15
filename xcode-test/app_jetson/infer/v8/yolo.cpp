#pragma once
#include "infer/v8/yolo.hpp"
static Logger gLogger;

extern int get_model_decrypt_value(unsigned char* encrypt,unsigned char* modelvalue,long len_);
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// extern int modleClassNum;

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.6
#define BBOX_CONF_THRESH 0.3
// #define NUM_CLASS modleClassNum

#define __V8__



static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}


// #ifdef __V8__
// static void generate_yolo_proposals(float* feat_blob,std::vector<std::vector<float>>& picked_proposals, int output_size, float prob_threshold, std::vector<Object>& objects)
static void generate_yolo_proposals(float* feat_blob, int output_size, float prob_threshold, std::vector<Object>& objects,int modleClassNum)
{ // V5 [1,39375,37] V8 [1,37 34000]
    const int num_class = modleClassNum;
    int d_dim = (num_class + 4);
    auto dets = output_size / d_dim;
	cv::Mat out1 = cv::Mat(d_dim, dets, CV_32F, feat_blob);
    // auto dets = 39375;
    for (int boxs_idx = 0; boxs_idx < dets; boxs_idx++)
    {
        // const int basic_pos = boxs_idx *(num_class + 5 + 32);
        // x1 y1 x2 y2 cls 32
        // float x1 = feat_blob[(dets*0)+boxs_idx];
        // float y1 = feat_blob[(dets*1)+boxs_idx];
        // // float w = feat_blob[basic_pos+2];
        // // float h = feat_blob[basic_pos+3];
        // float x2 = feat_blob[(dets*2)+boxs_idx];
        // float y2 = feat_blob[(dets*3)+boxs_idx];

        
        float w = out1.at<float>(2, boxs_idx);  //w
        float h = out1.at<float>(3, boxs_idx);  //h
        float x1 = (out1.at<float>(0, boxs_idx)-0.5*w);  //cx
        float y1 = (out1.at<float>(1, boxs_idx)-0.5*h);  //cy

        // float box_objectness = feat_blob[basic_pos+4];
        // std::cout<<*feat_blob<<std::endl;
        for (int class_idx = 0; class_idx < num_class; class_idx++)
        {
            float box_cls_score = out1.at<float>(4+class_idx, boxs_idx);
            float box_prob = box_cls_score;//box_objectness * box_cls_score;
            if (box_prob > prob_threshold)
            {
                std::vector<float> temp_proto;
                for(int protos = num_class + 4;protos<d_dim;protos++){
                    temp_proto.push_back(out1.at<float>(protos, boxs_idx));
                }
                Object obj;
                obj.rect.x = x1;
                obj.rect.y = y1;
                obj.rect.width = w;
                obj.rect.height = h;
                obj.label = class_idx;
                obj.prob = box_prob;
                obj.mask_feat = temp_proto;
                objects.push_back(obj);
                // std::vector<float> temp_proto(feat_blob + basic_pos + 5 + num_class, feat_blob + basic_pos  + num_class + 5 + 32);
				// picked_proposals.push_back(temp_proto);
            }

        } // class loop
    }
}
// #endif



static void decode_outputs(float* prob, int output_size, std::vector<BoxInfo>& outputs, float scale, const int img_w, const int img_h,int modleClassNum) {
        std::vector<Object> proposals;
        generate_yolo_proposals(prob, output_size, BBOX_CONF_THRESH, proposals,modleClassNum);
        // std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

        qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, NMS_THRESH);


        int count = picked.size();

        // std::cout << "num of boxes: " << count << std::endl;

        std::vector<Object> objects(count);
        outputs.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x) / scale;
            float y0 = (objects[i].rect.y) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;

            outputs[i].x1 = x0;
            outputs[i].x2 = x1;
            outputs[i].y1 = y0;
            outputs[i].y2 = y1;
            outputs[i].label = objects[i].label;
            outputs[i].score = objects[i].prob;
        }

        
}



// static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, std::string f)
// {
//     static const char* class_names[] = {
//             "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//             "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//             "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//             "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//             "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//             "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//             "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//             "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
//             "hair drier", "toothbrush"
//         };

//     cv::Mat image = bgr.clone();

//     for (size_t i = 0; i < objects.size(); i++)
//     {
//         const Object& obj = objects[i];

//         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

//         cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
//         float c_mean = cv::mean(color)[0];
//         cv::Scalar txt_color;
//         if (c_mean > 0.5){
//             txt_color = cv::Scalar(0, 0, 0);
//         }else{
//             txt_color = cv::Scalar(255, 255, 255);
//         }

//         cv::rectangle(image, obj.rect, color * 255, 2);

//         char text[256];
//         sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

//         int baseLine = 0;
//         cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

//         cv::Scalar txt_bk_color = color * 0.7 * 255;

//         int x = obj.rect.x;
//         int y = obj.rect.y + 1;
//         //int y = obj.rect.y - label_size.height - baseLine;
//         if (y > image.rows)
//             y = image.rows;
//         //if (x + label_size.width > image.cols)
//             //x = image.cols - label_size.width;

//         cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
//                       txt_bk_color, -1);

//         cv::putText(image, text, cv::Point(x, y + label_size.height),
//                     cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
//     }

//     cv::imwrite("det_res.jpg", image);
//     fprintf(stderr, "save vis file\n");
//     /* cv::imshow("image", image); */
//     /* cv::waitKey(0); */
// }



YOLO::YOLO(std::string engine_file_path,int w ,int h):
INPUT_W(w),
INPUT_H(h)
{
    size_t size{0};
    static int devId = 0;
    cudaSetDevice(devId);
    char *trtModelStream{nullptr};
    
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);

        unsigned char* modelvalue;
        get_model_decrypt_value((unsigned char*)trtModelStream,modelvalue,size);

        file.close();
    }else{
        std::cout << "engine read file error: "<<engine_file_path << std::endl;
    }
    std::cout << "engine init finished: "<<engine_file_path << std::endl;
    // static IRuntime* runtime;
    // if(!runtime)
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);

    
    assert(engine != nullptr); 
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    // const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    // std::cout << "-----"<<outputIndex<< std::endl;
    
 
    // const int outputIndex2 = engine->getBindingIndex("442");
    // std::cout << "-----"<<outputIndex2<< std::endl;

    // int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    // auto in_dims = engine->getBindingDimensions(inputIndex);
    // in_dims.d[0] = 1;
    // in_dims.d[2] = 1024;
    // in_dims.d[3] = 1024;



    #if NV_TENSORRT_MAJOR == 10
    inputIndex = 0;
    outputIndex = 1;
    auto    in_dims = engine->getTensorShape(INPUT_BLOB_NAME);
    auto    out_dims = engine->getTensorShape(OUTPUT_BLOB_NAME);
    #else   
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    auto   in_dims = engine->getBindingDimensions(inputIndex);
    auto    out_dims = engine->getBindingDimensions(outputIndex);
    #endif

    INPUT_W = in_dims.d[2];
    INPUT_H = in_dims.d[3];

    // context->setBindingDimensions(inputIndex,in_dims);
    // const int32_t data[4] = {1,3,1024,1024};
    // context->setInputShapeBinding(inputIndex,data);


    // auto out_dims = engine->getBindingDimensions(outputIndex);
    for(int j=0;j<out_dims.nbDims;j++) {
        this->output_size *= out_dims.d[j];
        //  std::cout << "-----"<<out_dims.d[j]<< std::endl;
    }
// #if 0
//     // modleClassNum = out_dims.d[1] - 4;
//     cv::Mat img = cv::Mat::zeros(1024,1024,CV_8UC3);
//     this->prob = new float[1024*1024];
//     float scale;
//     doInference(*context, img, this->prob, 1024*1024, img.size() ,scale);
// #endif

    modleClassNum = out_dims.d[1] - 4;
    this->prob = new float[this->output_size];

    // engine->destroy();
}



YOLO::~YOLO()
{
    std::cout<<"yolo destroy"<<std::endl;
    #if NV_TENSORRT_MAJOR == 10    
    delete this->context;
    delete this->engine;
    delete this->runtime;
    #else
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    #endif
    delete[] this->prob;
}



std::vector<BoxInfo> YOLO::detect_img(cv::Mat img)
{
    // cv::Mat img = cv::imread(image_path);
    int img_w = img.cols;
    int img_h = img.rows;
    auto start1 = std::chrono::system_clock::now();
    cv::Mat pr_img = this->static_resize(img);
    auto end1 = std::chrono::system_clock::now();

    // std::cout << "blob image" << std::endl;
    // float* blob;
    // blob = blobFromImage(pr_img);
    // float scale = std::min(this->INPUT_W / (img.cols*1.0), this->INPUT_H / (img.rows*1.0));

    // run inference
    auto start = std::chrono::system_clock::now();
    float scale = 0.;
    doInference(*context, pr_img, this->prob, output_size, pr_img.size(),scale);
    auto end = std::chrono::system_clock::now();
    scale = std::min(this->INPUT_W / (img.cols*1.0), this->INPUT_H / (img.rows*1.0));
    // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    auto start2 = std::chrono::system_clock::now();
    std::vector<BoxInfo> objects;
    decode_outputs(this->prob, this->output_size, objects, scale, img_w, img_h, modleClassNum);
    auto end2 = std::chrono::system_clock::now();
    // std::cout << "pre: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << "ms" << ", infer: "<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << ", post: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << "ms" << std::endl;
    // pre: 1ms, infer: 46ms, post: 0ms

    // draw_objects(img, objects, image_path);
    // delete prob;
    return objects;
}

cv::Mat YOLO::static_resize(cv::Mat& img) {
    float r = std::min(this->INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(this->INPUT_W, this->INPUT_H, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}

// float* YOLO::blobFromImage(cv::Mat& img){
//     cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

//     float* blob = new float[img.total()*3];
//     int channels = 3;
//     int img_h = img.rows;
//     int img_w = img.cols;
//     for (size_t c = 0; c < channels; c++) 
//     {
//         for (size_t  h = 0; h < img_h; h++) 
//         {
//             for (size_t w = 0; w < img_w; w++) 
//             {
//                 blob[c * img_w * img_h + h * img_w + w] =
//                     (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f);
//             }
//         }
//     }
//     return blob;
// }

void YOLO::doInference(IExecutionContext& context, cv::Mat img, float* output, const int output_size, cv::Size input_shape ,float &scale) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    // std::cout<<"__"<<engine.getNbBindings()<<std::endl;
    // assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    // const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);


    assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    // assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kINT8);
    // const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);
    // assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kINT8);
    // int mBatchSize = engine.getMaxBatchSize();

    // Create GPU buffers on device
    // CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    CHECK(cudaMalloc(&buffers[1], output_size*sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));




    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    std::vector<cv::Mat> channels;
	cv::split(img,channels);
    int row = img.rows;
    int col = img.cols;

	void* imgbuffer[3];
	CHECK(cudaMalloc((&imgbuffer[0]), row*col * sizeof(unsigned char)));
	CHECK(cudaMalloc((&imgbuffer[1]), row*col * sizeof(unsigned char)));
	CHECK(cudaMalloc((&imgbuffer[2]), row*col * sizeof(unsigned char)));

	CHECK(cudaMalloc((&buffers[0]), row*col*3 * sizeof(float)));

	CHECK(cudaMemcpyAsync((void*)(imgbuffer[0]), (void*)(channels[0].data), row*col * sizeof(unsigned char), cudaMemcpyHostToDevice,stream));
	CHECK(cudaMemcpyAsync((void*)(imgbuffer[1]), (void*)(channels[1].data), row*col * sizeof(unsigned char), cudaMemcpyHostToDevice,stream));
	CHECK(cudaMemcpyAsync((void*)(imgbuffer[2]), (void*)(channels[2].data), row*col * sizeof(unsigned char), cudaMemcpyHostToDevice,stream));
	// CHECK(cudaMemcpy((void*)(dfbuffer[1]), (void*)(&df2), 1 * sizeof(float), cudaMemcpyHostToDevice));


	MutiFun((unsigned char*)imgbuffer[0], (float*)buffers[0],row*col,0);
	MutiFun((unsigned char*)imgbuffer[1], (float*)(buffers[0]),row*col,row*col);
	MutiFun((unsigned char*)imgbuffer[2], (float*)(buffers[0]),row*col,row*col*2);

	cudaFree(imgbuffer[0]);
	cudaFree(imgbuffer[1]);
	cudaFree(imgbuffer[2]);
    scale = std::min(this->INPUT_W / (img.cols*1.0), this->INPUT_H / (img.rows*1.0));




    
    #if NV_TENSORRT_MAJOR == 10
    // 这里开始，需要进行输入输出的地址注册
    context.setInputTensorAddress(engine.getIOTensorName(inputIndex), buffers[inputIndex]);
    context.setOutputTensorAddress(engine.getIOTensorName(outputIndex), buffers[outputIndex]);
    // 接下来就是推理部分，这里不需要放缓存了
    context.enqueueV3(stream);
    // 完成推理将缓存地址中输出数据移动出来，后面也是和旧版本一样了
    #else
    context.enqueueV2(&buffers[inputIndex], stream, nullptr);
    #endif
    CHECK(cudaMemcpyAsync(output, buffers[1], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    // CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[1]));
    CHECK(cudaFree(buffers[0]));
}


