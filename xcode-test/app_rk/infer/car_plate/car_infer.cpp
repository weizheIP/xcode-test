#include "car_infer.h"
#include "common/license/license.h"
#include <algorithm>

using namespace std;


static void dump_tensor_attr_(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}
static unsigned char *load_data(FILE *fp, size_t ofst, size_t sz)
{
    unsigned char *data;
    unsigned char *outdata;
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
    int nret = get_model_decrypt_value(data,outdata,sz);
    if(0 != nret)
    {
        printf("decode model failure.\n");
        return NULL;
    }
    
    return data;
}
static unsigned char* load_model_(const char* filename, int* model_size)
{
    FILE *fp;
    unsigned char *data;

    fp = fopen(filename, "rb");
    if (NULL == fp)
    {
        printf("Open file %s failed.\n", filename);
        return NULL;
    }
    printf("Open model file %s .\n", filename);
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}
//------------------------------------------------------------------------------------------
static int rknn_GetTop_(float* pfProb, float* pfMaxProb, uint32_t* pMaxClass, uint32_t outputCount, uint32_t topNum)
{
  uint32_t i, j;

#define MAX_TOP_NUM 20
  if (topNum > MAX_TOP_NUM)
    return 0;

  memset(pfMaxProb, 0, sizeof(float) * topNum);
  memset(pMaxClass, 0xff, sizeof(float) * topNum);

  for (j = 0; j < topNum; j++) {
    for (i = 0; i < outputCount; i++) {
      if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
          (i == *(pMaxClass + 4))) {
        continue;
      }

      if (pfProb[i] > *(pfMaxProb + j)) {
        *(pfMaxProb + j) = pfProb[i];
        *(pMaxClass + j) = i;
      }
    }
  }

  return 1;
}

template<class ForwardIterator>
inline static size_t argmax_(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}
//----------------------------------------------------------------------------------------
static double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static int saveFloat(const char* file_name, float* output, int element_size)
{
  FILE* fp;
  fp = fopen(file_name, "w");
  for (int i = 0; i < element_size; i++) {
    fprintf(fp, "%.6f\n", output[i]);
  }
  fclose(fp);
  return 0;
}

cv::Mat resize_image(cv::Mat srcimg, int *newh, int *neww, int *top, int *left)
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

//----------------------------------------------------------------------------------------
CarPlateRec::CarPlateRec(char* modelPath):
_modelInWidth(),
_modelInHeight(),
_modelInChannels(),
_modelPath(modelPath)
{
    _keys = {  "#", 
                "A", "B", "C", "D", "E", "F", "G", 
                "H", "J", "K", "L", "M", "N", 
                "P", "Q", "R", "S", "T", 
                "U", "V", "W", "X", "Y", "Z", 
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
                 "京","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","皖","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新","学","港","澳","警","挂","使","领","民","航"
            };
}

CarPlateRec::~CarPlateRec()
{
    if (_ctx > 0)
    {
        rknn_destroy(_ctx);
    }
    if (_model) {
        free(_model);
    }
}

int CarPlateRec::init(int coreIndex)
{
    
    _model = load_model_(_modelPath, &_modelLen);

    int ret = rknn_init(&_ctx, _model, _modelLen, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        free(_model);
        _model = nullptr;
        return -1;
    }
    free(_model);
    _model = 0;

    // Get Model Input Output Info
    // rknn_input_output_num io_num;
    ret = rknn_query(_ctx, RKNN_QUERY_IN_OUT_NUM, &_ioNum, sizeof(_ioNum));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    // printf("model input num: %d, output num: %d\n", _ioNum.n_input, _ioNum.n_output);

    rknn_tensor_attr input_attrs[_ioNum.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < _ioNum.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        // dump_tensor_attr(&(input_attrs[i]));
    }

    // printf("output tensors:\n");
    rknn_tensor_attr output_attrs[_ioNum.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < _ioNum.n_output; i++) {
        output_attrs[i].index = i;
            ret = rknn_query(_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        // dump_tensor_attr(&(output_attrs[i]));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        // printf("model is NCHW input fmt\n");
        _modelInChannels = input_attrs[0].dims[1];
        _modelInHeight  = input_attrs[0].dims[2];
        _modelInWidth   = input_attrs[0].dims[3];
        // printf("[Info] carPlate model %d %d %d\n", _modelInWidth, _modelInHeight,_modelInChannels);
    } else {
        // printf("model is NHWC input fmt\n");
        _modelInHeight  = input_attrs[0].dims[1];
        _modelInWidth   = input_attrs[0].dims[2];
        _modelInChannels = input_attrs[0].dims[3];
        // printf("[Info] carPlate model %d %d %d\n", _modelInWidth, _modelInHeight,_modelInChannels);
    }


    // coreIndex =0;//3
    rknn_core_mask core_mask;
    switch (coreIndex)
    {
        case 1:
            core_mask = RKNN_NPU_CORE_0;
            break;
        case 2:
            core_mask = RKNN_NPU_CORE_1;
            break;
        case 3:
            core_mask = RKNN_NPU_CORE_2;
            break;
    }
    if(coreIndex)
    {
        int ret = rknn_set_core_mask(_ctx, core_mask);
    }

    return 0;
}

static inline bool isNumOrLetter(char word)
{
    if((word <=59 && word>=48) || (word<=90 && word>=65) || (word<=122 && word>=97))
        return 1;
    else
        return 0;
}

static inline std::string check_chinese_car_plate_(const std::string& carPlate)
{
    //在UTF-8编码格式下一个中文三个字节
    //7位 油车(3+6,9字节) 8位(3+7，10字节) 电车
    //最后一位有可能是中文
    if(carPlate.size() != 9 && carPlate.size() != 10)
    {
        //最后一位是中文
        if(carPlate.size() == 11 && (carPlate[10]>59 && carPlate[10]<48) && (carPlate[10]>90 && carPlate[10]<65) && (carPlate[10]>122 && carPlate[10]<97))
        {
            /*
                '学','港','澳','警','挂','使','领'
                这些字一定是最后一个字
            */
            bool isRight = false;
            std::vector<std::string> specialWorld = {"学","港","澳","警","挂","使","领"};
            for(int i = 0; i < specialWorld.size(); i++)
            {
                if(carPlate.find(specialWorld[i]) != std::string::npos)
                {
                    isRight = true;
                    break;
                }
            }
            if(!isRight)
                return "";
        }
        else
        {
            // std::cout<<"[INFO] carPalte length is not right:"<<carPlate<<",len:"<<carPlate.size()<<"\n";
            return "";
        }
    }
    //如果第二位不是英文，过滤
    if(!(carPlate[3] >= 65 && carPlate[3] <= 90)) //!= A-Z
    {
        return "";
    }
    /*
        如果第三四五六七位不是英文数字过滤
    */
    if(isNumOrLetter(carPlate[4]) && isNumOrLetter(carPlate[5]) && isNumOrLetter(carPlate[6]) && isNumOrLetter(carPlate[7]) && isNumOrLetter(carPlate[8]))
    {}
    else
    {
        return "";
    }
    
    return carPlate;
}

std::string CarPlateRec::get_car_plate_code(cv::Mat& org_img)
{
    string ans;
    cv::Mat img = org_img;
    if (org_img.cols != _modelInWidth || org_img.rows != _modelInHeight) {
        // printf("resize %d %d to %d %d\n", img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
        // cv::resize(orig_img, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), 0, 0, cv::INTER_LINEAR);
        cv::resize(org_img, img, cv::Size(_modelInWidth, _modelInHeight));
    }

    

    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].size  = img.cols * img.rows * img.channels() * sizeof(uint8_t);
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].buf   = img.data;

    int ret = rknn_inputs_set(_ctx, _ioNum.n_input, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return ans;
    }

    // Run
    // printf("rknn_run\n");
    ret = rknn_run(_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return ans;
    }

    // Get Output
    rknn_output outputs[_ioNum.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < _ioNum.n_output; i++) {
        outputs[i].want_float = 1;
    }

    ret = rknn_outputs_get(_ctx, _ioNum.n_output, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return ans;
    }

    float*   outputData = (float*)outputs[0].buf;
    uint32_t sz     = outputs[0].size / 4;
    // cout << "num:" << sz << endl;
    
    int w = 76;
    
    std::string strRes;
    std::vector<float> scores;
    int lastIndex = 0;
    int maxIndex;
    float maxValue;
    for (int i = 0; i < 25; i++) {
        maxIndex = int(argmax_(&outputData[i * w], &outputData[(i + 1) * w - 1]));
        maxValue = float(*std::max_element(&outputData[i * w], &outputData[(i + 1) * w - 1]));
        if (maxIndex > 0 && maxIndex < w && (!(i > 0 && maxIndex == lastIndex))) {
            scores.emplace_back(maxValue);
            strRes.append(_keys[maxIndex]);
        }
        lastIndex = maxIndex;
    }


    // cout << "strRes:" << strRes << endl;

    rknn_outputs_release(_ctx, 1, outputs);
    
    ans = check_chinese_car_plate_(strRes);

    return ans;
}

//---------------------------------------------------------
extern int anchor0[6];
extern int anchor1[6];
extern int anchor2[6];
CarPlateDet::CarPlateDet(char* modelPath,int type):
_modelInWidth(),
_modelInHeight(),
_modelInChannels(),
_modelPath(modelPath),
_NMS_THRESH(0.1),
_BOX_THRESH(0.5)
{
    if(type)
    {
        //anchor0 = {10, 13, 16, 30, 33, 23};
        //anchor1 = {30, 61, 62, 45, 59, 119};
        //anchor2 = {116, 90, 156, 198, 373, 326};
        anchor0[0] = 10;
        anchor0[1] = 13;
        anchor0[2] = 16;
        anchor0[3] = 30;
        anchor0[4] = 33;
        anchor0[5] = 23;
        anchor1[0] = 30;
        anchor1[1] = 61;
        anchor1[2] = 62;
        anchor1[3] = 45;
        anchor1[4] = 59;
        anchor1[5] = 119;
        anchor2[0] = 116;
        anchor2[1] = 90;
        anchor2[2] = 156;
        anchor2[3] = 198;
        anchor2[4] = 373;
        anchor2[5] = 326;
    }
    
}

CarPlateDet::~CarPlateDet()
{
    if (_ctx > 0)
    {
        rknn_destroy(_ctx);
    }
    if (_model) {
        free(_model);
    }
}

int CarPlateDet::init(int coreIndex)
{
    _model = load_model_(_modelPath, &_modelLen);

    int ret = rknn_init(&_ctx, _model, _modelLen, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        free(_model);
        return -1;
    }
    free(_model);
    _model = 0;

    // Get Model Input Output Info
    // rknn_input_output_num io_num;
    ret = rknn_query(_ctx, RKNN_QUERY_IN_OUT_NUM, &_ioNum, sizeof(_ioNum));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    // printf("model input num: %d, output num: %d\n", _ioNum.n_input, _ioNum.n_output);

    rknn_tensor_attr input_attrs[_ioNum.n_input];

    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < _ioNum.n_input; i++) {
        input_attrs[i].index = i;
        ret                  = rknn_query(_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
        }
        // dump_tensor_attr(&(input_attrs[i]));
    }

    

    int channel = 3;
    int width   = 0;
    int height  = 0;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        // printf("model is NCHW input fmt\n");
        _modelInChannels = input_attrs[0].dims[1];
        _modelInHeight  = input_attrs[0].dims[2];
        _modelInWidth   = input_attrs[0].dims[3];
    } else {
        
        _modelInHeight  = input_attrs[0].dims[1];
        _modelInWidth   = input_attrs[0].dims[2];
        _modelInChannels = input_attrs[0].dims[3];
        // printf("model is NHWC input fmt w:%d,h:%d.\n",_modelInWidth,_modelInHeight);
    }

    // int coreIndex =0; //2
    rknn_core_mask core_mask;
    switch (coreIndex)
    {
        case 1:
            core_mask = RKNN_NPU_CORE_0;
            break;
        case 2:
            core_mask = RKNN_NPU_CORE_1;
            break;
        case 3:
            core_mask = RKNN_NPU_CORE_2;
            break;
    }
    if(coreIndex)
    {
        int ret = rknn_set_core_mask(_ctx, core_mask);
    }

    return 0;
}

int CarPlateDet::detect(cv::Mat &img,std::vector<CarBoxInfo>& boxes)
{
    struct timeval start_time, stop_time;
    const float    nms_threshold      = _NMS_THRESH;
    const float    box_conf_threshold = _BOX_THRESH;

    

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index        = 0;
    inputs[0].type         = RKNN_TENSOR_UINT8;
    inputs[0].size         = _modelInWidth * _modelInHeight * _modelInChannels;
    inputs[0].fmt          = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    cv::Mat img2;
    cv::cvtColor(img, img2, cv::COLOR_BGR2RGB);
    int img_width  = img.cols;
    int img_height = img.rows;

    int newh = 0, neww = 0, padh = 0, padw = 0;
    
    img2 = resize_image(img2, &newh, &neww, &padh, &padw);

    inputs[0].buf = (void*)img2.data;
    gettimeofday(&start_time, NULL);
    rknn_inputs_set(_ctx, _ioNum.n_input, inputs);

    rknn_output outputs[_ioNum.n_output];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < _ioNum.n_output; i++) {
        // outputs[i].want_float = 0;
        outputs[i].want_float = 1;
    }
    int ret = rknn_run(_ctx, NULL);
    ret = rknn_outputs_get(_ctx, _ioNum.n_output, outputs, NULL);
    gettimeofday(&stop_time, NULL);
    // printf("once run use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // post process
    float scale_w = (float)_modelInWidth / img_width;
    float scale_h = (float)_modelInHeight / img_height;

    detect_result_group_t detect_result_group;
    std::vector<float>    out_scales;
    std::vector<int32_t>  out_zps;

    rknn_tensor_attr output_attrs[_ioNum.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < _ioNum.n_output; i++) {
        output_attrs[i].index = i;
        ret                   = rknn_query(_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        // dump_tensor_attr(&(output_attrs[i]));
    }

    for (int i = 0; i < _ioNum.n_output; ++i) {
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }
    post_process((float*)outputs[0].buf, (float*)outputs[1].buf, (float*)outputs[2].buf, _modelInHeight, _modelInWidth,
                box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

    float ratioh = (float)img.rows / newh, ratiow = (float)img.cols / neww;
    for (int i = 0; i < detect_result_group.count; i++) 
    {
        detect_result_t* det_result = &(detect_result_group.results[i]);
        int x1 = (det_result->box.left  -padw)* ratiow;
        int y1 = (det_result->box.top  -padh)* ratioh;
        int x2 = (det_result->box.right -padw)* ratiow;
        int y2 = (det_result->box.bottom  -padh)* ratioh;
        std::vector<float>  landmark;

        // printf("borod not check:%d %d %d %d\n",x1,y1,x2,y2);

        x1 = std::max(std::min((float)x1, (float)(img.cols - 1)), 0.f);
        y1 = std::max(std::min((float)y1, (float)(img.rows - 1)), 0.f);
        // x2 = std::max(std::min((float)x2, (float)(img_width - 1)), 0.f);
        // y2 = std::max(std::min((float)y2, (float)(img_height - 1)), 0.f);
        
        x2 = std::max(std::min((float)x2, (float)(img.cols - 1)), 0.f);
        y2 = std::max(std::min((float)y2, (float)(img.rows - 1)), 0.f);
        
        // printf("borod check::%d %d %d %d\n",x1,y1,x2,y2);
        
        for (int i=0; i<4;i++){
            int landmark_x = (det_result->landmark[i*2]-padw)* ratiow;
            int landmark_y = (det_result->landmark[i*2+1]-padh)* ratioh;
            landmark.push_back(landmark_x);
            landmark.push_back(landmark_y);
            // cv::circle(orig_img, cv::Point(landmark_x, landmark_y), 2, cv::Scalar(0,0,255),-1);
        }
        CarBoxInfo box;
        box.x1 = x1;
        box.y1 = y1;
        box.x2 = x2;
        box.y2 = y2;



        box.score = det_result->prop;
        box.label = det_result->name;
        box.landmark = landmark;
        boxes.push_back(box);
    }

    ret = rknn_outputs_release(_ctx, _ioNum.n_output, outputs);

    return 0;
}

//----------------------------------------------------------------------------------------

CarPlateColor::CarPlateColor(char* modelPath):
_modelInWidth(),
_modelInHeight(),
_modelInChannels(),
_modelPath(modelPath)
{

}

CarPlateColor::~CarPlateColor()
{
    if (_ctx > 0)
    {
        rknn_destroy(_ctx);
    }
    if (_model) {
        free(_model);
    }
}

int CarPlateColor::init()
{
    _model = load_model_(_modelPath, &_modelLen);
    int ret = rknn_init(&_ctx, _model, _modelLen, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        free(_model);
        return -1;
    }
    free(_model);
    _model = 0;

    // Get Model Input Output Info
    // rknn_input_output_num io_num;
    ret = rknn_query(_ctx, RKNN_QUERY_IN_OUT_NUM, &_ioNum, sizeof(_ioNum));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    // printf("model input num: %d, output num: %d\n", _ioNum.n_input, _ioNum.n_output);


    rknn_tensor_attr input_attrs[_ioNum.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < _ioNum.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        // dump_tensor_attr(&(input_attrs[i]));
    }

    // printf("output tensors:\n");
    rknn_tensor_attr output_attrs[_ioNum.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < _ioNum.n_output; i++) {
        output_attrs[i].index = i;
            ret = rknn_query(_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        // dump_tensor_attr(&(output_attrs[i]));
    }

    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        // printf("model is NCHW input fmt\n");
        _modelInChannels = input_attrs[0].dims[1];
        _modelInHeight  = input_attrs[0].dims[2];
        _modelInWidth   = input_attrs[0].dims[3];
        //  printf("[Info] carPlate model %d %d %d\n", _modelInWidth, _modelInHeight,_modelInChannels);
    } else {
        // printf("model is NHWC input fmt\n");
        _modelInHeight  = input_attrs[0].dims[1];
        _modelInWidth   = input_attrs[0].dims[2];
        _modelInChannels = input_attrs[0].dims[3];
        //  printf("[Info] carPlate model %d %d %d\n",  _modelInWidth, _modelInHeight,_modelInChannels);
    }


    return 0;
}

int CarPlateColor::get_car_plate_color(cv::Mat& org_img)
{

    cv::Mat img;

    cv::cvtColor(org_img, img, cv::COLOR_BGR2RGB);

    if (org_img.cols != _modelInWidth || org_img.rows != _modelInHeight) {
        // printf("resize %d %d to %d %d\n", img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
        // cv::resize(orig_img, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), 0, 0, cv::INTER_LINEAR);
        cv::resize(img, img, cv::Size(_modelInWidth, _modelInHeight));
    }

    
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].size  = img.cols * img.rows * img.channels() * sizeof(uint8_t);
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].buf   = img.data;

    int ret = rknn_inputs_set(_ctx, _ioNum.n_input, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    ret = rknn_run(_ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }

    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret                   = rknn_outputs_get(_ctx, 1, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }

    for (int i = 0; i < _ioNum.n_output; i++) {
        uint32_t MaxClass[5];
        float    fMaxProb[5];
        float*   buffer = (float*)outputs[i].buf;
        uint32_t sz     = outputs[i].size / 4;

        int len = sizeof(buffer) / sizeof(buffer[0]);
        // printf("%3d: %3d: \n", outputs[i].size,len);

        // cout << "malloc_usable_size(arr):" << malloc_usable_size(buffer) / sizeof(*buffer) << endl;


        rknn_GetTop_(buffer, fMaxProb, MaxClass, sz, 5);

        // printf(" --- Top5 ---\n");
        // for (int i = 0; i < 5; i++) {
        // printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);      
        // }

        return MaxClass[0];
        
        // printf(" --- buffer ---\n");
        //   for (int i = 0; i < sz; i++) {
        //   printf(" %8.6f \n", buffer[i]);
        // }
    }
    return -1;

}




