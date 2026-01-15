
#include "face_recognition.h"
#include "common/license/license.h"
using namespace std;
using namespace cv;

static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    int ret;

    data = NULL;

    if (NULL == fp)
    {
        return NULL;
    }

    ret = fseek(fp, ofst, SEEK_SET);
    if (ret != 0)
    {
        printf("blob seek failure.\n");
        return NULL;
    }

    data = (unsigned char *)malloc(sz);
    if (data == NULL)
    {
        printf("buffer malloc failure.\n");
        return NULL;
    }
    ret = fread(data, 1, sz, fp);

    unsigned char *outdata;
    int nret = get_model_decrypt_value(data,outdata,sz);
    if(0 != nret)
    {
        printf("decode model failure.\n");
        return NULL;
    }
    return data;
}

static unsigned char *load_model(const char *filename, int *model_size)
{

    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

int FaceRecognizer::init(){
        /* Create the neural network */
    printf("Loading mode...\n");
    int model_data_size = 0;
    unsigned char *model_data = load_model(model_name, &model_data_size);
    int ret = rknn_init(&ctx, model_data, model_data_size, 1, NULL);

    free(model_data);
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }

    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version,
                    sizeof(rknn_sdk_version));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("sdk version: %s driver version: %s\n", version.api_version,
        version.drv_version);

    
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0)
    {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input,
        io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                        sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("[INFO] rknn_init error ret=%d\n", ret);
            return -1;
        }
        // dump_tensor_attr(&(input_attrs[i]));
    }
    // g_model_w = input_attrs[0].dims[1];
    // g_model_h = input_attrs[0].dims[2];


    return 0;
}





FaceRecognizer::~FaceRecognizer(){
    int ret = rknn_destroy(ctx);
    if(ret == RKNN_SUCC)
    {
        std::cout<< "RKNN release success!"<<std::endl;
    }
    else{
        std::cout<< "RKNN release false!"<<std::endl;
    }
    // delete resize_buf;
    // resize_buf = nullptr;
}



std::vector<float>  FaceRecognizer::feature(cv::Mat orig_img)
{
    // std::vector<BoxInfo> bbox(200,0);
    // if(orig_img.empty())return -1;

    int status = 0;
    // char *model_name = NULL;
    size_t actual_size = 0;
    int img_width = 0;
    int img_height = 0;
    int img_channel = 0;
    struct timeval start_time, stop_time;
    int ret;

    // // init rga context
    // rga_buffer_t src;
    // rga_buffer_t dst;
    // im_rect src_rect;
    // im_rect dst_rect;
    // memset(&src_rect, 0, sizeof(src_rect));
    // memset(&dst_rect, 0, sizeof(dst_rect));
    // memset(&src, 0, sizeof(src));
    // memset(&dst, 0, sizeof(dst));

    // if (argc != 3)
    // {
    //     printf("Usage: %s <rknn model> <jpg> \n", argv[0]);
    //     return -1;
    // }

    // printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n",
    //     box_conf_threshold, nms_threshold);

    
    // cv::Mat orig_img;
    // cap.read(orig_img);
    // cv::Mat orig_img = cv::imread("1.jpg");
    cv::Mat img_,img;
    cv::cvtColor(orig_img, img_, cv::COLOR_BGR2RGB);
    cv::resize(img_, img, cv::Size(112,112));
    // int newh = 0, neww = 0, padh = 0, padw = 0;
    // cv::Mat img = resize_image(img_, &newh, &neww, &padh, &padw);

    img_width = img.cols;
    img_height = img.rows;
    // printf("img width = %d, img height = %d\n", img_width, img_height);
    // printf("model input num: %d, output num: %d\n", io_num.n_input,
        // io_num.n_output);

    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++)
    {
        input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),
                        sizeof(rknn_tensor_attr));
        if (ret < 0)
        {
            printf("[INFO] rknn_init error ret=%d\n", ret);
            // return -1;
        }
        // dump_tensor_attr(&(input_attrs[i]));
    }

    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        output_attrs[i].index = i;  
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]),
                        sizeof(rknn_tensor_attr));
        // dump_tensor_attr(&(output_attrs[i]));
    }
    // class_num = (output_attrs[0].dims[1]/3) - 5;

    int channel = 3;
    int width = 0;
    int height = 0;
  
    // printf("model is NHWC input fmt\n");
    height = input_attrs[0].dims[1];
    width = input_attrs[0].dims[2];
    channel = input_attrs[0].dims[3];



    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = width * height * channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    
    inputs[0].buf = img.data;//&(data_vec[0]);//resize_buf;
    gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++)
    {
        outputs[i].want_float = 1;
    }

    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    gettimeofday(&stop_time, NULL);
    // std::cout<<"once run use "<<(__get_us(stop_time) - __get_us(start_time)) / 1000<<" ms"<<std::endl;
    
    //post process
    float scale_w = (float)width / orig_img.cols;
    float scale_h = (float)height / orig_img.rows;

    std::vector<float> out_scales;
    std::vector<int32_t> out_zps;
    for (int i = 0; i < io_num.n_output; ++i)
    {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    
    // featureMap = cv::Mat(1,output_attrs[0].dims[1],CV_32FC1,));
    float* ptr=(float*)(outputs[0].buf);
    std::vector<float> vec(ptr, ptr + output_attrs[0].dims[1]);

    return vec;
}

