#include "face2yolov5.h"
#include "common/license/license.h"

using namespace ns_yoloface;
// static void dump_tensor_attr(rknn_tensor_attr* attr)
// {
//   printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
//          "zp=%d, scale=%f\n",
//          attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
//          attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
//          get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
// }

// double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static unsigned char* load_data(FILE* fp, size_t ofst, size_t sz)
{
  unsigned char* data;
  int            ret;

  data = NULL;

  if (NULL == fp) {
    return NULL;
  }

  ret = fseek(fp, ofst, SEEK_SET);
  if (ret != 0) {
    printf("blob seek failure.\n");
    return NULL;
  }

  data = (unsigned char*)malloc(sz);
  if (data == NULL) {
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

static unsigned char* load_model(const char* filename, int* model_size)
{
  FILE*          fp;
  unsigned char* data;

  fp = fopen(filename, "rb");
  if (NULL == fp) {
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

static cv::Mat resize_image(cv::Mat srcimg, int *newh, int *neww, int *top, int *left)
{
    int inpHeight=640;
    int inpWidth=640;
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = inpHeight;
	*neww = inpWidth;
	cv::Mat dstimg;
	if (srch != srcw)
	{
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1)
		{
			*newh = inpHeight;
			*neww = int(inpWidth / hw_scale);
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*left = int((inpWidth - *neww) * 0.5);
			cv::copyMakeBorder(dstimg, dstimg, 0, 0, *left, inpWidth - *neww - *left, cv::BORDER_CONSTANT, 114);
		}
		else
		{
			*newh = (int)inpHeight * hw_scale;
			*neww = inpWidth;
			cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
			*top = (int)(inpHeight - *newh) * 0.5;
			cv::copyMakeBorder(dstimg, dstimg, *top, inpHeight - *newh - *top, 0, 0, cv::BORDER_CONSTANT, 114);
		}
	}
	else
	{
		cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
	}
	return dstimg;
}

int Face_Detector2::init()
{
    /* Create the neural network */
    printf("Loading mode...\n");
    int            model_data_size = 0;
    unsigned char* model_data      = load_model(model_name, &model_data_size);
    int ret                        = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    free(model_data);
    rknn_sdk_version version;
    ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("rk 2 sdk version: %s driver version: %s\n", version.api_version, version.drv_version);
    
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
    return 0; //不返回int在release模式下会没有初始值会报错
}

int Face_Detector2::detect_face(cv::Mat orig_img,std::vector<Yolov5Face_BoxStruct> &face_result,float nms_threshold,float box_conf_threshold)
{
    int ret = 0;
    struct timeval start_time, stop_time;
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            printf("rknn_init error ret=%d\n", ret);
            return -1;
        }
        // dump_tensor_attr(&(input_attrs[i]));
    }
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        // dump_tensor_attr(&(output_attrs[i]));
    }
    int channel = 3;
    int width   = 0;
    int height  = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        // printf("model is NCHW input fmt\n");
        channel = input_attrs[0].dims[1];
        height  = input_attrs[0].dims[2];
        width   = input_attrs[0].dims[3];
    } else {
        // printf("model is NHWC input fmt\n");
        height  = input_attrs[0].dims[1];
        width   = input_attrs[0].dims[2];
        channel = input_attrs[0].dims[3];
    }

    // printf("model input height=%d, width=%d, channel=%d\n", height, width, channel);
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index        = 0;
    inputs[0].type         = RKNN_TENSOR_UINT8;
    inputs[0].size         = width * height * channel;
    inputs[0].fmt          = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;

    // 加letterbox
    cv::Mat img;
    cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
    int img_width  = img.cols;
    int img_height = img.rows;
    // printf("img width = %d, img height = %d\n", img_width, img_height);
    int newh = 0, neww = 0, padh = 0, padw = 0;
    img = resize_image(img, &newh, &neww, &padh, &padw);
    inputs[0].buf = (void*)img.data;
    // cv::imwrite("resize_input.jpg", img);

    gettimeofday(&start_time, NULL);
    rknn_inputs_set(ctx, io_num.n_input, inputs);

    rknn_output outputs[io_num.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < io_num.n_output; i++) {
        outputs[i].want_float = 1;
    }
    ret = rknn_run(ctx, NULL);
    ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
    gettimeofday(&stop_time, NULL);
    // printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
    float scale_w = (float)width / img_width;
    float scale_h = (float)height / img_height;

    detect_result_group_t detect_result_group;
    std::vector<float>    out_scales;
    std::vector<int32_t>  out_zps;
    for (int i = 0; i < io_num.n_output; ++i) {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    post_process((float*)outputs[0].buf, (float*)outputs[1].buf, (float*)outputs[2].buf, height, width,
               box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    float ratioh = (float)orig_img.rows / newh, ratiow = (float)orig_img.cols / neww;
    char text[256];
    for (int i = 0; i < detect_result_group.count; i++) {
        detect_result_t* det_result = &(detect_result_group.results[i]);
        int x1 = (det_result->box.left  -padw)* ratiow;
        int y1 = (det_result->box.top  -padh)* ratioh;
        int x2 = (det_result->box.right -padw)* ratiow;
        int y2 = (det_result->box.bottom  -padh)* ratioh;
        std::vector<float>  landmark = det_result->landmark;

        Yolov5Face_BoxStruct yolofs;
        yolofs.x1 = std::max(std::min((float)x1, (float)(orig_img.cols - 1)), 0.f);
        yolofs.y1 = std::max(std::min((float)y1, (float)(orig_img.rows - 1)), 0.f);
        yolofs.x2 = std::max(std::min((float)x2, (float)(orig_img.cols - 1)), 0.f);
        yolofs.y2 = std::max(std::min((float)y2, (float)(orig_img.rows - 1)), 0.f);

        if(yolofs.x2 -yolofs.x1 <=3 || yolofs.y2 - yolofs.y1 <=3) //上下边缘
            continue;

        yolofs.score = det_result->prop;
        for (int i=0; i<5;i++){
            int landmark_x = (landmark[i*2]-padw)* ratiow;
            int landmark_y = (landmark[i*2+1]-padh)* ratioh;
            yolofs.keypoint.push_back(cv::Point(landmark_x , landmark_y));
        }

        face_result.push_back(yolofs);
    }
    // cout<<face_result.size()<<"--count"<<endl;
    ret = rknn_outputs_release(ctx, io_num.n_output, outputs);
    return 0;
}

Face_Detector2::~Face_Detector2()
{
    int ret = rknn_destroy(ctx);
    if(ret == RKNN_SUCC)
    {
        std::cout<< "RKNN release success!"<<std::endl;
    }
    else{
        std::cout<< "RKNN release false!"<<std::endl;
    }
}