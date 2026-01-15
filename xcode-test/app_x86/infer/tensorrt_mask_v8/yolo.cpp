#pragma once
#include "infer/tensorrt_mask_v8/yolo.hpp"
#include <chrono>

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
#define BBOX_CONF_THRESH 0.1
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


#if 1 
static void generate_yolo_proposals(float* feat_blob,std::vector<std::vector<float>>& picked_proposals, int output_size, float prob_threshold, std::vector<Object>& objects)
{ // V5 [1,39375,37] V8 [1,37 34000]
    const int num_class = NUM_CLASS;
    int d_dim = (num_class + 4 + 32);
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
#endif


cv::Mat MatMut_Format(cv::Mat dim1_m,cv::Mat dim2_m)
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
    MatMuti((float*)dim1,(float*)dim2,(float*)result,dims1,dims2);

    cv::Mat matmulRes(dim2_m.cols,dim1_m.rows,CV_32FC1);
    cudaMemcpy(matmulRes.data,result, dim1_rows*dim2_cols*sizeof(float), cudaMemcpyDeviceToHost);

// #if 0
//     float* show_result = (float*)malloc(sizeof(float)*dim1_rows*dim2_cols);
//     float* show_dim1 = (float*)malloc(sizeof(float)*dim1_rows*dim1_cols);
//     float* show_dim2 = (float*)malloc(sizeof(float)*dim2_rows*dim2_cols);
//     cudaMemcpy(show_result,result, dim1_rows*dim2_cols*sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(show_dim1,dim1, dim1_rows*dim1_cols*sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(show_dim2,dim2, dim2_rows*dim2_cols*sizeof(float), cudaMemcpyDeviceToHost);



//     float* show_md = (float*)md.data;
//     float* show_md_result = (float*)matmulRes.data;
//     for(int j =0;j<dim1_rows*dim2_cols;j++)
//     {
//         // for(int i = j*1024;i<j*1024+1024;i++){
//             printf("index--%d,%.2f,",j,show_md[j]);
//             printf("%.2f\n",show_md_result[j]);
//         // }
//     }
//     free(show_result);
//     free(show_dim1);
//     free(show_dim2);
// #endif

    
    // memcpy(matmulRes.data,show_result,dim2_m.cols*dim1_m.rows*sizeof(float));

    
    cudaFree(dim1);
    cudaFree(dim2);
    cudaFree(result);
    return matmulRes;
}


// static void decode_outputs(float* prob, float* maskData, int output_size, std::vector<Object>& objects, float scale, const int img_w, const int img_h) {

void Restore_LetterboxImage(const cv::Mat &src, cv::Mat &dst,float pad_w,float pad_h,float scale)
{
    cv::Mat rec_img = src(cv::Rect(pad_w,pad_h,src.cols-2*pad_w,src.rows-2*pad_h));
    
    float out_w = rec_img.cols/scale;
    float out_h = rec_img.rows/scale;

    cv::resize(rec_img, dst, cv::Size(out_w, out_h));
}





static void decode_outputs(float* prob, float* maskData, int output_size, std::vector<Object>& objects, float pad_w, float pad_h, float scale, const int img_w, const int img_h) {
   
    const int modle_size = 1280;
    std::vector<std::vector<float>> picked_proposals; //mask
    std::vector<std::vector<float>> temp_mask_proposals;

    auto start = std::chrono::steady_clock::now();
    std::vector<Object> proposals;
    generate_yolo_proposals(prob,picked_proposals, output_size, BBOX_CONF_THRESH, proposals);
    // std::cout << "num of boxes before nms: " << proposals.size()<<" mask: "<<picked_proposals.size() << std::endl;

    qsort_descent_inplace(proposals);//这里导致mask的顺序和box对不上，用cv::dnn::NMSBoxes就没这个问题得4.0以上版本

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    // std::cout << "time nms: "  << elapsed_seconds.count() << std::endl;  
    
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

// #if 1
        float x0 = (objects[i].rect.x - pad_w) / scale;  //  /scale;
        float y0 = (objects[i].rect.y - pad_h) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - pad_w) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - pad_h) / scale;
// #endif

// #if 0
//         // 掐头去尾 各3%
//         float d_part_per = 0.03;
//         float x0 ,x1,y0,y1;
//         float per_w = objects[i].rect.width * d_part_per;
//         float per_h = objects[i].rect.height * d_part_per;
//         if(objects[i].rect.width >= objects[i].rect.height * 2)
//         {
//             x0 = (objects[i].rect.x + per_w - pad_w) / scale;  //  /scale;
//             y0 = (objects[i].rect.y - pad_h) / scale;
//             x1 = (objects[i].rect.x + objects[i].rect.width - per_w - pad_w) / scale;
//             y1 = (objects[i].rect.y + objects[i].rect.height - pad_h) / scale;
//         }
//         else if(objects[i].rect.width * 2 <= objects[i].rect.height){
            
//             x0 = (objects[i].rect.x - pad_w) / scale;  //  /scale;
//             y0 = (objects[i].rect.y + per_h - pad_h) / scale;
//             x1 = (objects[i].rect.x + objects[i].rect.width - pad_w) / scale;
//             y1 = (objects[i].rect.y + objects[i].rect.height - per_h - pad_h) / scale;
//         }
//         else{
//             x0 = (objects[i].rect.x + per_w - pad_w) / scale;  //  /scale;
//             y0 = (objects[i].rect.y + per_h - pad_h) / scale;
//             x1 = (objects[i].rect.x + objects[i].rect.width - per_w - pad_w) / scale;
//             y1 = (objects[i].rect.y + objects[i].rect.height - per_h - pad_h) / scale;
//         }

// #endif

        objects[i].rect_160.x=int(std::max(std::min(objects[i].rect.x/4, (modelsize_4 - 1)), 0));
        objects[i].rect_160.y=int(std::max(std::min(objects[i].rect.y/4, (modelsize_4 - 1)), 0));
        objects[i].rect_160.width=int(std::max(std::min(objects[i].rect.width/4, (modelsize_4 - objects[i].rect.x/4 - 1)), 0));    
        objects[i].rect_160.height=int(std::max(std::min(objects[i].rect.height/4, (modelsize_4 - objects[i].rect.y/4 - 1)), 0));    


        // objects[i].rect_160.x=int(std::max(std::min(objects[i].rect.x, (modelsize_4 - 1)), 0));
        // objects[i].rect_160.y=int(std::max(std::min(objects[i].rect.y, (modelsize_4 - 1)), 0));
        // objects[i].rect_160.width=int(std::max(std::min(objects[i].rect.width, (modelsize_4 - objects[i].rect.x - 1)), 0));    
        // objects[i].rect_160.height=int(std::max(std::min(objects[i].rect.height, (modelsize_4 - objects[i].rect.y - 1)), 0));    

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
	cv::Mat maskProposals;
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
		maskProposals.push_back( cv::Mat(temp_mask_proposals[i]).t() );
    

	float* pdata = maskData;
	std::vector<float> mask(pdata, pdata + 32 * modelsize_4 * modelsize_4);
	
	cv::Mat mask_protos = cv::Mat(mask);
	// cv::Mat protos = mask_protos.reshape(0, { 32,160 * 160 });//将prob1的值 赋给mask_protos
    
	cv::Mat protos = mask_protos.reshape(0, 32);//将prob1的值 赋给mask_protos
    
   
	auto start_mask = std::chrono::steady_clock::now();
    // cv::Mat matmulRes2 = (maskProposals * protos).t();//n*32 32*25600 A*B是以数学运算中矩阵相乘的方式实现的，要求A的列数等于B的行数时 160 * 160=25600
    cv::Mat matmulRes = MatMut_Format(maskProposals,protos);
    auto end_mask = std::chrono::steady_clock::now();
	

    // cv::Mat masks = matmulRes.reshape(objects.size(), { 160,160 });
	cv::Mat masks = matmulRes.reshape(objects.size(), modelsize_4);
	// std::cout << protos.size()<<"----"<< masks.size()<<masks.channels() << endl;
	std::vector<cv::Mat> maskChannels;
	cv::split(masks, maskChannels);
    // std::cout << "mask num: "<<maskChannels.size()<<std::endl;;
    // cv::Rect holeImgRect(0, 0, img_w, img_h);
    


	for (int i = 0; i < objects.size(); ++i) {
		cv::Mat dest, mask, dest160 ;
		//sigmoid
		cv::exp(-maskChannels[i], dest);
		dest = 1.0 / (1.0 + dest);//160*160
        // std::cout <<"--------"<< dest<<endl;
        dest160=dest.clone();
        // cv::resize(dest160,dest160,cv::Size(modle_size,modle_size));
        // // cv::imwrite("resule.jpg",dest160);
        // Restore_LetterboxImage(dest160,dest160, pad_w, pad_h, scale);
        // cv::imwrite("rest_resule.jpg",dest160);
		cv::Rect temp_rect160 = objects[i].rect_160 ;

        cv::Mat mask160 = dest160(temp_rect160) > MASK_THRESH;               
        int nCount = cv::countNonZero(mask160);       
        objects[i].area = nCount;




        // float d_part_per = 0.10;
        // float per_w = objects[i].rect.width * d_part_per;
        // float per_h = objects[i].rect.height * d_part_per;

        // cv::Rect temp_rect160_2 = objects[i].rect ;
        // if(temp_rect160_2.width >= temp_rect160_2.height * 2/3)
        // {
        //     temp_rect160_2.width = (temp_rect160_2.width - per_w );
        // }
        // else if(temp_rect160_2.width * 2/3 <= temp_rect160_2.height){
        //     temp_rect160_2.height = (temp_rect160_2.height - per_h );
        // }
        // else{
        //     temp_rect160_2.width = (temp_rect160_2.width - per_w ) ;
        //     temp_rect160_2.height = (temp_rect160_2.height - per_h ) ;
        // }

		// cv::Mat mask160_2 = dest160(temp_rect160_2) > MASK_THRESH;    
        // if(mask160_2.cols == 0 || mask160_2.rows == 0){
        //     objects[i].boxMask = mask160_2;
        //     objects[i].area = 0;
        //    continue;
        // }    
        // int nCount_2 = cv::countNonZero(mask160_2);


        // if(temp_rect160.width >= temp_rect160.height * 2/3)
        // {
        //     temp_rect160.x = (temp_rect160.x + per_w);  //  /scale;
        //     temp_rect160.width = (temp_rect160.width - per_w);  //  /scale;
        // }
        // else if(temp_rect160.width * 2/3 <= temp_rect160.height){
        //     temp_rect160.y = (temp_rect160.y + per_h);
        //     temp_rect160.height = (temp_rect160.height - per_h);  //  /scale;
        // }
        // else{
        //     temp_rect160.x = (temp_rect160.x + per_w);  //  /scale;
        //     temp_rect160.y = (temp_rect160.y + per_h);
        //     temp_rect160.width = (temp_rect160.width - per_w);  //  /scale;
        //     temp_rect160.height = (temp_rect160.height - per_h);  //  /scale;
        // }

		// cv::Mat mask160_1 = dest160(temp_rect160) > MASK_THRESH;    
        // if(mask160_1.cols == 0 || mask160_1.rows == 0){
        //     objects[i].boxMask = mask160_1;
        //     objects[i].area = 0;
        //    continue;
        // }    
        // int nCount_1 = cv::countNonZero(mask160_1);

        

        // if(nCount_1 > nCount_2)
        // {
        //     objects[i].rect = temp_rect160;
        //     objects[i].boxMask = mask160_1;
		//     objects[i].area = nCount_1;
        // }
        // else{
        //     objects[i].rect = temp_rect160_2;
        //     objects[i].boxMask = mask160_2;
		//     objects[i].area = nCount_2;
        // }

        // break;
	}
    end = std::chrono::steady_clock::now();
    // std::cout << "time infer post: "  << std::chrono::duration_cast<std::chrono::milliseconds>(end_mask - start_mask).count()<<"--"<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start_mask).count() << std::endl; 
   
}




void DrawPred(cv::Mat& img, std::vector<Object> result) {
	cv::Mat mask = img.clone();
    // cv::Mat Binary = cv::Mat::zeros(img.rows,img.cols,cv::CV_8UC1);
    cv::Scalar color(0,0,255);
    cv::Scalar color_text(0,255,0);
    // cv::Scalar color_binary(255);
	for (int i = 0; i < result.size(); i++) {
		// int left, top;
		// if(result[i].prob < 0.6) continue;
        int c_x = result[i].rect.x + result[i].rect.width/2;
		int c_y = result[i].rect.y + result[i].rect.height/2;
		int color_num = i;
		cv::rectangle(img, result[i].rect, color, 2, 8);
		mask(result[i].rect).setTo(color, result[i].boxMask);
        // mask.setTo(color, result[i].boxMask);
        // Binary(result[i].rect).setTo(color_binary, result[i].boxMask);
		// string label = classNames[result[i].id] + ":" + to_string(result[i].confidence);
		int baseLine;
		// Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		// top = max(top, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
        char f_d[20] = {0};
        snprintf(f_d,20,"%.2f",result[i].prob);
        cv::putText(img, std::to_string(result[i].area), cv::Point(c_x, c_y), cv::FONT_HERSHEY_SIMPLEX, 1, color_text, 2);
        // cv::putText(img, f_d, cv::Point(c_x, c_y), cv::FONT_HERSHEY_SIMPLEX, 1, color_text, 2);
	}
    // cv::imwrite("save.jpg",Binary);
	cv::addWeighted(img, 0.5, mask, 0.5, 0, img); //add mask to src
	// imshow("1", img);
	// imwrite("out.jpg", img);
	// waitKey();
	//destroyAllWindows();

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
    cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);

    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
    int left = (static_cast<int>(out_w) - mid_w) / 2;
    int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
    return pad_info;
}




YOLO::YOLO(std::string engine_file_path)
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
    }
    std::cout << "engine init finished" << std::endl;

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size);

    
    assert(engine != nullptr); 
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    int intputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    auto in_dims = engine->getBindingDimensions(intputIndex);

    // 设置输入尺寸
    INPUT_W = in_dims.d[2];
    INPUT_H = in_dims.d[3];

    int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    // std::cout << "-----index:"<<outputIndex<< std::endl;
    

//     const int outputIndex2 = engine->getBindingIndex("442");
//  std::cout << "-----"<<outputIndex2<< std::endl;

    auto out_dims = engine->getBindingDimensions(outputIndex);
    for(int j=0;j<out_dims.nbDims;j++) {
        this->output_size *= out_dims.d[j];
        //  std::cout << "-----dim size:"<<out_dims.d[j]<< std::endl;
    }
    


    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME_MASK);
    // std::cout << "-----index:"<<outputIndex<< std::endl;
    out_dims = engine->getBindingDimensions(outputIndex);
    for(int j=0;j<out_dims.nbDims;j++) {
        this->output_size_mask *= out_dims.d[j];
        //  std::cout << "-----dim size:"<<out_dims.d[j]<< std::endl;
    }

    for(int i =0;i<queue_det_list.max_size_;i++)
    {
        Det_Struct _tmp_struct;
        _tmp_struct.prob_mask = new float[this->output_size_mask]();
        _tmp_struct.prob = new float[this->output_size]();
        queue_det_list.Push(_tmp_struct);
    }


    InitInfer(*context);
    // engine->destroy();
}


YOLO::~YOLO()
{
    bRun_Decode = false;
    if(th_Decode != nullptr && th_Decode->joinable()){
        th_Decode->join();
    }
    std::cout<<"yolo destroy"<<std::endl;
    this->context->destroy();
    this->engine->destroy();
    this->runtime->destroy();
    while(queue_det_list.GetSize()!=0){
        Det_Struct _tmp;
        queue_det_list.Pop(_tmp);
        delete _tmp.prob;
        delete _tmp.prob_mask;
    }
    while(queue_result_list.GetSize()!=0){
        Det_Struct _tmp;
        queue_result_list.Pop(_tmp);
        delete _tmp.prob;
        delete _tmp.prob_mask;
    }
    while(queue_det_ForDet_list.GetSize()!=0){
        Det_Struct _tmp;
        queue_det_ForDet_list.Pop(_tmp);
        delete _tmp.prob;
        delete _tmp.prob_mask;
    }
}



std::vector<Object> YOLO::detect_img(cv::Mat& img)
{
    // img = cv::imread("/home/nvidia/zhihuigongdi/webapi_demo/alg_demo/run/pig2.jpg");
    
    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img;// = this->static_resize(img);
    std::vector<float> pad_info = LetterboxImage(img,pr_img,cv::Size(INPUT_W,INPUT_H));
    // std::cout << "blob image" << std::endl;

    // float* blob;
    // blob = blobFromImage(pr_img);
    // float scale = std::min(this->INPUT_W / (img.cols*1.0), this->INPUT_H / (img.rows*1.0));

    // run inference
    float scale = 0.;
    auto start = std::chrono::system_clock::now();
    float *prob = new float[this->output_size]();
    float *prob_mask = new float[this->output_size_mask]();
    doInference(*context, pr_img, prob,prob_mask, output_size, pr_img.size(),scale);
    auto det_end = std::chrono::system_clock::now();
    std::vector<Object> objects;
    decode_outputs(prob,prob_mask, output_size, objects, pad_info[0], pad_info[1], pad_info[2], img.cols, img.rows);  
#ifdef DRAW_MASK
    DrawPred(img,objects);   
#endif
    auto end = std::chrono::system_clock::now();
    std::cout 
    << "Det Time:" << std::chrono::duration_cast<std::chrono::milliseconds>(det_end - start).count() << "ms"
    <<",Run Time:"<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // decode_outputs(this->prob, this->output_size, objects, scale, img_w, img_h);
    // draw_objects(img, objects, image_path);
    // delete blob;
    delete prob;
    delete prob_mask;
    return objects;
}


Det_Struct YOLO::detect_img(cv::Mat& img,bool& bGet)
{
    auto start = std::chrono::system_clock::now();
    bGet = false;
    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img;
    std::vector<float> pad_info = LetterboxImage(img,pr_img,cv::Size(INPUT_W,INPUT_H));
    // run inference
    float scale = 0.;
    Det_Struct result;
    Det_Struct _queue_det;
    if(queue_result_list.GetSize()!=0 || queue_det_list.GetSize() == 0)
    {
        while(queue_result_list.GetSize() == 0)usleep(1000);
        Det_Struct _result_tmp;
        queue_result_list.Pop(_result_tmp);
        result.img = _result_tmp.img.clone();
        result.objects = _result_tmp.objects;
        // queue_det_list.Push(_result_tmp);
        // _result_tmp.objects.clear();
        // _result_tmp.img.release();
        _queue_det = _result_tmp;
        bGet = true;
    }
    else 
    {
        queue_det_list.Pop(_queue_det);
    }
    
    _queue_det.img = img;
    _queue_det.pad_info = pad_info;
    // _queue_det.prob = new float[this->output_size]();
    // _queue_det.prob_mask = new float[this->output_size_mask]();

    doInference(*context, pr_img, _queue_det.prob,_queue_det.prob_mask, output_size, pr_img.size(),scale);
    if(queue_det_ForDet_list.GetSize()!=queue_det_ForDet_list.max_size_)
        queue_det_ForDet_list.Push(_queue_det);
    else queue_det_list.Push(_queue_det);

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
                cv::Mat img = queue_det.img;
                queue_det.objects.clear();
                decode_outputs(queue_det.prob,queue_det.prob_mask, output_size, queue_det.objects, queue_det.pad_info[0], queue_det.pad_info[1], queue_det.pad_info[2], img.cols, img.rows);   
#ifdef DRAW_MASK
                DrawPred(queue_det.img,queue_det.objects);
#endif
                // delete queue_det.prob;
                // delete queue_det.prob_mask;
                if(queue_result_list.GetSize() == queue_result_list.max_size_)
                {
                    // usleep(1000);
                    queue_det_list.Pop(queue_det);
                    continue;
                }
                else queue_result_list.Push(queue_det);
                
            }
            
        });
    }

    
    
    auto det_end = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    std::cout
    <<"Run Time:"<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    return result;
}



void YOLO::InitInfer(IExecutionContext& context)
{
    // engine = context.getEngine();
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    assert(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    // assert(engine->getBindingDataType(inputIndex) == nvinfer1::DataType::kINT8);
    outputIndex_det = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(engine->getBindingDataType(outputIndex_det) == nvinfer1::DataType::kFLOAT);

    outputIndex_seg = engine->getBindingIndex(OUTPUT_BLOB_NAME_MASK);
    assert(engine->getBindingDataType(outputIndex_seg) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = engine->getMaxBatchSize();

    CHECK(cudaMalloc(&buffers[inputIndex], 3 * INPUT_W * INPUT_H * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex_seg], output_size_mask*sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex_det], output_size*sizeof(float)));

    CHECK(cudaStreamCreate(&stream));

    CHECK(cudaMalloc((&imgbuffer[0]), INPUT_W*INPUT_H * sizeof(unsigned char)));
	CHECK(cudaMalloc((&imgbuffer[1]), INPUT_W*INPUT_H * sizeof(unsigned char)));
	CHECK(cudaMalloc((&imgbuffer[2]), INPUT_W*INPUT_H * sizeof(unsigned char)));

}

void YOLO::doInference(IExecutionContext& context, cv::Mat img, float* output,float* output_mask, const int output_size, cv::Size input_shape ,float &scale) {
    // const ICudaEngine& engine = context.getEngine();

    // // Pointers to input and output device buffers to pass to engine.
    // // Engine requires exactly IEngine::getNbBindings() number of buffers.
    // // std::cout<<"__"<<engine.getNbBindings()<<std::endl;
    // // assert(engine.getNbBindings() == 2);
    // void* buffers[3];

    // // In order to bind the buffers, we need to know the names of the input and output tensors.
    // // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    // const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);

    // assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);
    // // assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kINT8);
    // int outputIndex_det = engine.getBindingIndex(OUTPUT_BLOB_NAME);
    // assert(engine.getBindingDataType(outputIndex_det) == nvinfer1::DataType::kFLOAT);

    // int outputIndex_seg = engine.getBindingIndex(OUTPUT_BLOB_NAME_MASK);
    // assert(engine.getBindingDataType(outputIndex_seg) == nvinfer1::DataType::kFLOAT);
    // // assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kINT8);
    // int mBatchSize = engine.getMaxBatchSize();

    // // Create GPU buffers on device
    // CHECK(cudaMalloc(&buffers[inputIndex], 3 * input_shape.height * input_shape.width * sizeof(float)));
    // CHECK(cudaMalloc(&buffers[outputIndex_seg], output_size_mask*sizeof(float)));
    // CHECK(cudaMalloc(&buffers[outputIndex_det], output_size*sizeof(float)));

    // // Create stream
    // cudaStream_t stream;
    // CHECK(cudaStreamCreate(&stream));

    


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


	MutiFun((unsigned char*)imgbuffer[0], (float*)buffers[inputIndex],row*col,0);
	MutiFun((unsigned char*)imgbuffer[1], (float*)(buffers[inputIndex]),row*col,row*col);
	MutiFun((unsigned char*)imgbuffer[2], (float*)(buffers[inputIndex]),row*col,row*col*2);

	// cudaFree(imgbuffer[0]);
	// cudaFree(imgbuffer[1]);
	// cudaFree(imgbuffer[2]);
    scale = std::min(this->INPUT_W / (img.cols*1.0), this->INPUT_H / (img.rows*1.0));


#if 0
    float* blob = new float[row*col*3];
    int channels_ = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels_; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    CHECK(cudaMemcpy(buffers[inputIndex],blob, 3 * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyHostToDevice));
#endif




    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    // CHECK(cudaMemcpyAsync(buffers[inputIndex], input, 3 * input_shape.height * input_shape.width * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(1, buffers, stream, nullptr);
    // context.enqueueV2(&buffers[inputIndex], stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex_det], output_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK(cudaMemcpyAsync(output_mask, buffers[outputIndex_seg], output_size_mask * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

#if 0
    float* blob = new float[row*col*3];
    float* result = new float[row*col*3];
    int channels_ = 3;
    int img_h = img.rows;
    int img_w = img.cols;
    for (size_t c = 0; c < channels_; c++) 
    {
        for (size_t  h = 0; h < img_h; h++) 
        {
            for (size_t w = 0; w < img_w; w++) 
            {
                blob[c * img_w * img_h + h * img_w + w] =
                    (((float)img.at<cv::Vec3b>(h, w)[c]) / 255.0f);
            }
        }
    }
    CHECK(cudaMemcpy(result, buffers[inputIndex], 3 * INPUT_W * INPUT_H * sizeof(float), cudaMemcpyDeviceToHost));
    for(int j =333*1200;j<3 * INPUT_W * INPUT_H-100;j+=100)
    {    
        int jjj = 0;
        for(int i = 0;i<100;i++){
            printf("input-%d:%.3f,%.3f\n",j+i,blob[j+i],result[j+i]);
        }
    }
    delete blob;
    delete result;
#endif
    // Release stream and buffers
    // cudaStreamDestroy(stream);
    // CHECK(cudaFree(buffers[inputIndex]));
    // CHECK(cudaFree(buffers[outputIndex_det]));
    // CHECK(cudaFree(buffers[outputIndex_seg]));
}

