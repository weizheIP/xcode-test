#include "infer/tensorrt_mask_v8_que/yolo_que.hpp"
#include <chrono>

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

// int modleClassNum = 1;

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.6
#define MASK_THRESH 0.5
#define BBOX_CONF_THRESH 0.3
#define NUM_CLASS 1


using namespace std;


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


static void generate_yolo_proposals(float* feat_blob,std::vector<std::vector<float>>& picked_proposals, int output_size, float prob_threshold, std::vector<Object>& objects)
{ // V5 [1,39375,37] V8 [1,37 34000]
    const int num_class = NUM_CLASS;
    int d_dim = (num_class + 4 + 32);
    auto dets = output_size / d_dim;
	cv::Mat out1 = cv::Mat(d_dim, dets, CV_32F, feat_blob).clone();
    // auto dets = 39375;
    for (int boxs_idx = 0; boxs_idx < dets; boxs_idx++)
    {                
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


cv::Mat MatMut_Format(cv::Mat dim1_m,cv::Mat dim2_m,int dx)
{
    int dim1_rows = dim1_m.rows;
    int dim1_cols = dim1_m.cols;
    int dim2_rows = dim2_m.rows;
    int dim2_cols = dim2_m.cols;
    // int dim2_2 = 102400;
    // cv::Mat Mdim1 = cv::Mat::eye(3,5,CV_32FC1);    
    cv::Mat Mdim1 = dim1_m.clone();
    cv::Mat Mdim2 = dim2_m.clone();
    // cv::Mat md = (Mdim1*Mdim2).t();
    Mdim2 = Mdim2.t();
    void* dim1,*dim2,*result;
    cudaMalloc(&dim1,dim1_rows*dim1_cols*sizeof(float));
    cudaMalloc(&dim2,dim2_rows*dim2_cols*sizeof(float));
    cudaMalloc(&result,dim1_rows*dim2_cols*sizeof(float));

    cudaMemset(result, 0, dim1_rows*dim2_cols*sizeof(float)); 
    cudaMemcpy(dim1,Mdim1.data, dim1_rows*dim1_cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dim2,Mdim2.data, dim2_rows*dim2_cols*sizeof(float), cudaMemcpyHostToDevice);    
    
    

    int dims1[2] = {dim1_rows,dim1_cols};
    int dims2[2] = {dim2_rows,dim2_cols};

    

    MatMuti((float*)dim1,(float*)dim2,(float*)result,dims1,dims2,dx);

    cv::Mat matmulRes(dim2_m.cols,dim1_m.rows,CV_32FC1);
    cudaMemcpy(matmulRes.data,result, dim1_rows*dim2_cols*sizeof(float), cudaMemcpyDeviceToHost);

    
    // memcpy(matmulRes.data,show_result,dim2_m.cols*dim1_m.rows*sizeof(float));

    
    cudaFree(dim1);
    cudaFree(dim2);
    cudaFree(result);
    return matmulRes;
}






static void decode_outputs(float* prob,float* maskData, cv::Mat& maskProposals,cv::Mat& protos, int output_size, std::vector<Object>& objects, float pad_w, float pad_h, float scale, const int img_w, const int img_h,int dx,int wh) {
    const int modle_size = wh;
    
    std::vector<std::vector<float>> picked_proposals; //mask
    std::vector<std::vector<float>> temp_mask_proposals;

    auto start = std::chrono::steady_clock::now();
    std::vector<Object> proposals;
    generate_yolo_proposals(prob,picked_proposals, output_size, BBOX_CONF_THRESH, proposals);
    // std::cout << "num of boxes before nms: " << proposals.size()<<" mask: "<<picked_proposals.size() << std::endl;

    qsort_descent_inplace(proposals);//这里导致mask的顺序和box对不上，用cv::dnn::NMSBoxes就没这个问题得4.0以上版本

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);
    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;
    // std::cout << "time nms: "  << elapsed_seconds.count() << std::endl;  //30只的时候5ms
    
    int count = picked.size();
    // int count = 1;
    if (count==0){
        return;
    }

    // std::cout << "num of boxes: " << count << std::endl;

    // float widthScale = (float)(img_w) / 640;
    // float heightScale = (float)(img_h) / 640;
    int modelsize_4=modle_size/4;
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        
        objects[i] = proposals[picked[i]];
        temp_mask_proposals.push_back(proposals[picked[i]].mask_feat);

        float x0 = (objects[i].rect.x - pad_w) / scale;  //  /scale;
        float y0 = (objects[i].rect.y - pad_h) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - pad_w) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - pad_h) / scale;

        objects[i].rect_160.x=int(std::max(std::min(objects[i].rect.x/4, (modelsize_4 - 1)), 0));
        objects[i].rect_160.y=int(std::max(std::min(objects[i].rect.y/4, (modelsize_4 - 1)), 0));
        objects[i].rect_160.width=int(std::max(std::min(objects[i].rect.width/4, (modelsize_4 - objects[i].rect.x/4 - 1)), 0));    
        objects[i].rect_160.height=int(std::max(std::min(objects[i].rect.height/4, (modelsize_4 - objects[i].rect.y/4 - 1)), 0));    

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = int(x0);
        objects[i].rect.y = int(y0);
        objects[i].rect.width = int(x1 - x0);
        objects[i].rect.height = int(y1 - y0);
        
        // std::cout << "boxes 1: " << x0 <<"-"<<y0 <<"-"<<x1<<"-" <<y1 << std::endl;
    }
    // 处理mask
    // cv::Mat maskProposals;
    for (int i = 0; i < temp_mask_proposals.size(); ++i){
        if(i<30)
            maskProposals.push_back( cv::Mat(temp_mask_proposals[i]).t() );        
    }

    float* pdata = maskData;
    std::vector<float> mask(pdata, pdata + 32 * modelsize_4 * modelsize_4);
    
    cv::Mat mask_protos = cv::Mat(mask).clone(); //注意内存会被释放
    // cv::Mat protos = mask_protos.reshape(0, { 32,160 * 160 });//将prob1的值 赋给mask_protos
    protos = mask_protos.reshape(0, 32);//将prob1的值 赋给mask_protos
}
    
static void decode_masks(const cv::Mat& maskProposals, const cv::Mat& protos, std::vector<Object>& objects,int modle_size) {

    if (objects.size()==0){
        sleep(0.01);
        return;
    }

    int modelsize_4=modle_size/4;
    auto start_mask = std::chrono::steady_clock::now();    
    // std::cout << protos.size()<<"----"<< maskProposals.size()<<"----"<< modelsize_4  << endl; // 25600*32 32*n

    // cv::Mat matmulRes2 = (maskProposals * protos).t();//n*32 32*25600 A*B是以数学运算中矩阵相乘的方式实现的，要求A的列数等于B的行数时 160 * 160=25600
    // 40T：cpu———— 7只时候52ms，22只时候180ms,50只320ms
    // 20T：cpu———— 19只时候99s
    // cv::Mat matmulRes =(maskProposals * protos).t();
    // cv::Mat matmulRes =(maskProposals * protos).t();
    // std::cout << "maskProposals: "<<maskProposals.rows<<"-"<<maskProposals.cols<<std::endl; //n*32
    // std::cout << "protos: "<<protos.rows<<"-"<<protos.cols<<std::endl; //32*102400
    // cv::Mat matmulRes;
    // cv::gemm(maskProposals, protos, 1, cv::noArray(), 0, matmulRes, cv::GEMM_2_T); // 使用GEMM函数，其中第三个参数是protos需要转置
    


    // 20T：cpu———— 9只55ms，19只时候99s
    // cv::Mat matmulRes =(maskProposals * protos).t();
    
    //！！！ 40T：cuda———— 40只的时候100ms，加上后面的145ms;但是检测稳定60ms;;  30只87--111ms;  1只75ms;  20只的时候共75-90ms,40只也是110ms，50只有时候到130ms。// 20T：cuda———— 2只time mask共120ms    
    cv::Mat matmulRes = MatMut_Format(maskProposals,protos,400); 


    // auto end_mask1 = std::chrono::steady_clock::now();
    // cv::Mat masks = matmulRes.reshape(objects.size(), { 160,160 });
    // 只要30只
    cv::Mat masks ;
    if(objects.size()<30)
        masks= matmulRes.reshape(objects.size(), modelsize_4);
    else
        masks = matmulRes.reshape(30, modelsize_4);
    // // std::cout << protos.size()<<"----"<< masks.size()<<masks.channels() << endl;
    std::vector<cv::Mat> maskChannels;
    cv::split(masks, maskChannels);
    

    int xx=0;
    for (int i = 0; i < objects.size(); ++i) {
        if(i<30){
            cv::Mat dest, mask, dest160 ;
            //sigmoid
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);//160*160
            // std::cout <<"--------"<< dest<<endl;
            dest160=dest.clone();
            // cv::resize(dest160,dest160,cv::Size(modle_size,modle_size));
            // cv::imwrite("resule.jpg",dest160);
            // Restore_LetterboxImage(dest160,dest160, pad_w, pad_h, scale);
            // cv::imwrite("rest_resule.jpg",dest160);
            // cv::Rect temp_rect160 = objects[i].rect ;
            cv::Rect temp_rect160 = objects[i].rect_160 ;

            cv::Mat mask160 = dest160(temp_rect160) > MASK_THRESH;    
            // if(mask160.cols == 0 || mask160.rows == 0){
            //     // objects[i].boxMask = mask160;
            //     objects[i].area = 0;
            // continue;
            // }    
            
            
            int nCount = cv::countNonZero(mask160);
            // objects[i].label = nCount;
            // std::cout << "nCount:" << nCount<<"--"<<temp_rect160.x << std::endl;

                // cv::resize(mask160,mask160,)
            // objects[i].boxMask = mask160;
            objects[i].area = nCount;
            xx=xx+nCount;
        }else{
            objects[i].area = xx/30;
        }
    }
    // end = std::chrono::steady_clock::now();
    auto end_mask = std::chrono::steady_clock::now();
    std::cout << "time post: "  <<  std::chrono::duration_cast<std::chrono::milliseconds>(end_mask - start_mask).count()<<"ms" << std::endl; 
    // 40T: 43只time post: 111--145ms
}


std::vector<float> LetterboxImage(const cv::Mat &src, cv::Mat &dst, const cv::Size &out_size)
{
    auto in_h = static_cast<float>(src.rows);
    auto in_w = static_cast<float>(src.cols);
    float out_h = out_size.height;
    float out_w = out_size.width;

    float scale = std::min(out_w / in_w, out_h / in_h);

    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);

    // cv::resize(src, dst, cv::Size(mid_w, mid_h));
    // dst = src;
    cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    
    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
    int left = (static_cast<int>(out_w) - mid_w) / 2;
    int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
    return pad_info;
}



YOLO_QUE::YOLO_QUE(std::string engine_file_path):
th_Decode(nullptr)
{
    size_t size{0};
    cudaSetDevice(0);
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
    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);

    
    assert(engine != nullptr); 
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    // int intputIndex = engine->getBindingIndex(INPUT_BLOB_NAME); //trt10有问题，用getNbIOTensors，getTensorShape 



    // nvinfer1::Dims in_dims,out_dims,out_dims2;
    #if NV_TENSORRT_MAJOR == 10
    inputIndex = 0;
    outputIndex_det = 1;
    outputIndex_seg = 2;
    auto    in_dims = engine->getTensorShape(INPUT_BLOB_NAME);
    auto    out_dims = engine->getTensorShape(OUTPUT_BLOB_NAME);
    auto    out_dims2 = engine->getTensorShape(OUTPUT_BLOB_NAME_MASK);
    #else    
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex_det = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    outputIndex_seg = engine->getBindingIndex(OUTPUT_BLOB_NAME_MASK);
    // cout<< inputIndex << " - " << outputIndex_det << " - " << outputIndex_seg << endl; //0-2-1 trt8 ?
    auto   in_dims = engine->getBindingDimensions(inputIndex);
    auto    out_dims = engine->getBindingDimensions(outputIndex_det);
    auto    out_dims2 = engine->getBindingDimensions(outputIndex_seg);
    #endif
    // 设置输入尺寸
    INPUT_W = in_dims.d[2];
    INPUT_H = in_dims.d[3];

    int cala_val = (INPUT_W / 4) * (INPUT_W / 4);

    // dx = INPUT_W / 4;
    wh = INPUT_W;
    for(int j=0;j<out_dims.nbDims;j++) {
        this->output_size *= out_dims.d[j];
        //  std::cout << "-----dim size:"<<out_dims.d[j]<< std::endl;
    }
    


    for(int j=0;j<out_dims2.nbDims;j++) {
        this->output_size_mask *= out_dims2.d[j];
        //  std::cout << "-----dim size:"<<out_dims.d[j]<< std::endl;
    }



    InitInfer(*context);
    // engine->destroy();
}


YOLO_QUE::~YOLO_QUE()
{
    bRun_Decode = false;
    if(th_Decode != nullptr && th_Decode->joinable()){
        th_Decode->join();
    }

    // if(th_LetterBox != nullptr && th_LetterBox->joinable()){
    //     th_LetterBox->join();
    // }
    cudaStreamDestroy(stream);    
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
   
}



// int indeximg = 1;
Det_Struct YOLO_QUE::detect_img(cv::Mat &img,bool& bGet)
{
    // indeximg++;
    // if(indeximg>5)indeximg = 1;
    auto start = std::chrono::system_clock::now();
    bGet = false;
    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img;
    std::vector<float> pad_info = LetterboxImage(img,pr_img,cv::Size(INPUT_W,INPUT_H));
    // run inference
    float scale = 0.;   
    auto letterBoxEnd = std::chrono::system_clock::now();

    Det_Struct _queue_det;
    _queue_det.img = img.clone();
   
    // 40T：检查60ms，nms等15ms；偶尔有波动会到100ms
    float *prob = new float[this->output_size]();
    float *prob_mask = new float[this->output_size_mask]();    
    auto det_start = std::chrono::system_clock::now();
    doInference(*context, pr_img, prob,prob_mask, output_size, pr_img.size(),scale);
    decode_outputs(prob,prob_mask,_queue_det.maskProposals, _queue_det.protos,output_size, _queue_det.objects, pad_info[0], pad_info[1], pad_info[2], img.cols, img.rows,dx, wh);   
    
    if(queue_det_ForDet_list.GetSize()!=queue_det_ForDet_list.max_size_)
        queue_det_ForDet_list.Push(_queue_det);
    delete prob; //数据用完再释放
    delete prob_mask;
    auto det_end = std::chrono::system_clock::now();

    if(th_Decode == nullptr){
        th_Decode = new std::thread([&](void){
            while(bRun_Decode)
            {   
                
                if(queue_det_ForDet_list.GetSize() == 0)
                {
                    usleep(1000);
                    continue;
                }
                Det_Struct queue_det;
                queue_det_ForDet_list.Pop(queue_det);              
                decode_masks(queue_det.maskProposals,queue_det.protos,  queue_det.objects,  wh);   
                queue_result_list.Push(queue_det);                
            }
            
        });
    }

    Det_Struct result;
    if(queue_result_list.GetSize()!=0)
    {
        while(queue_result_list.GetSize() == 0)usleep(1000);
        Det_Struct _result_tmp;
        queue_result_list.Pop(_result_tmp);
        result.img = _result_tmp.img.clone();
        result.objects = _result_tmp.objects;
        bGet = true;
    }

    
    auto end = std::chrono::system_clock::now();
    // if((runCnt_++ % 50) == 0)
    // {   
        std::cout<<"time all:"<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" 
        // <<",lettetBox:"<<std::chrono::duration_cast<std::chrono::milliseconds>(letterBoxEnd - start).count()
        <<" time det:"<<std::chrono::duration_cast<std::chrono::milliseconds>(det_end - det_start).count()<<"ms" << std::endl;
    // }
    

    return result;
}


void YOLO_QUE::InitInfer(IExecutionContext& context)
{
    // engine = context.getEngine();

    assert(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    // assert(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kINT8);
    assert(engine->getBindingDataType(outputIndex_det) == nvinfer1::DataType::kFLOAT);  
    assert(engine->getBindingDataType(outputIndex_seg) == nvinfer1::DataType::kFLOAT);
    // int mBatchSize = engine->getMaxBatchSize();
    

    CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_W * INPUT_H * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex_seg], output_size_mask*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex_det], output_size*sizeof(float)));

    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMalloc((&imgbuffer[0]), INPUT_W*INPUT_H * sizeof(unsigned char)));
	CHECK(cudaMalloc((&imgbuffer[1]), INPUT_W*INPUT_H * sizeof(unsigned char)));
	CHECK(cudaMalloc((&imgbuffer[2]), INPUT_W*INPUT_H * sizeof(unsigned char)));

}

void YOLO_QUE::doInference(IExecutionContext& context, cv::Mat img, float* output,float* output_mask, const int output_size, cv::Size input_shape ,float &scale) {
    
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB); 
    std::vector<cv::Mat> channels;
	cv::split(img,channels);
    int row = img.rows;
    int col = img.cols;

	// void* imgbuffer[3];
	// CHECK(cudaMalloc((&imgbuffer[0]), row*col * sizeof(unsigned char)));
	// CHECK(cudaMalloc((&imgbuffer[1]), row*col * sizeof(unsigned char)));
	// CHECK(cudaMalloc((&imgbuffer[2]), row*col * sizeof(unsigned char)));

    // 2和0颠倒，用于BGR->RGB
	CHECK(cudaMemcpyAsync((void*)(imgbuffer[0]), (void*)(channels[0].data), row*col * sizeof(unsigned char), cudaMemcpyHostToDevice,stream));
	CHECK(cudaMemcpyAsync((void*)(imgbuffer[1]), (void*)(channels[1].data), row*col * sizeof(unsigned char), cudaMemcpyHostToDevice,stream));
	CHECK(cudaMemcpyAsync((void*)(imgbuffer[2]), (void*)(channels[2].data), row*col * sizeof(unsigned char), cudaMemcpyHostToDevice,stream));
	// CHECK(cudaMemcpy((void*)(dfbuffer[1]), (void*)(&df2), 1 * sizeof(float), cudaMemcpyHostToDevice));


	MutiFun2((unsigned char*)imgbuffer[0], (float*)buffers[inputIndex],row*col,0);
	MutiFun2((unsigned char*)imgbuffer[1], (float*)(buffers[inputIndex]),row*col,row*col);
	MutiFun2((unsigned char*)imgbuffer[2], (float*)(buffers[inputIndex]),row*col,row*col*2);

	// cudaFree(imgbuffer[0]);
	// cudaFree(imgbuffer[1]);
	// cudaFree(imgbuffer[2]);
    scale = std::min(this->INPUT_W / (img.cols*1.0), this->INPUT_H / (img.rows*1.0));


    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    // CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    // context.enqueue(1, buffers, stream, nullptr);
    // 前面都是一样的

    #if NV_TENSORRT_MAJOR == 10
    // 这里开始，需要进行输入输出的地址注册
    context.setInputTensorAddress(engine->getIOTensorName(inputIndex), buffers[inputIndex]);
    context.setOutputTensorAddress(engine->getIOTensorName(outputIndex_det), buffers[outputIndex_det]);
    context.setOutputTensorAddress(engine->getIOTensorName(outputIndex_seg), buffers[outputIndex_seg]);
    // 接下来就是推理部分，这里不需要放缓存了
    context.enqueueV3(stream);
    // 完成推理将缓存地址中输出数据移动出来，后面也是和旧版本一样了
    #else
    context.enqueueV2(&buffers[inputIndex], stream, nullptr);
    #endif

    CHECK(cudaMemcpyAsync(output, buffers[outputIndex_det], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output_mask, buffers[outputIndex_seg], output_size_mask * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    // cudaStreamDestroy(stream);
    // CHECK(cudaFree(buffers[inputIndex]));
    // CHECK(cudaFree(buffers[outputIndex_det]));
    // CHECK(cudaFree(buffers[outputIndex_seg]));
}
