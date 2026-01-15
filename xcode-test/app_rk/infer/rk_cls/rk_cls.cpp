
#include "infer/rk_cls/rk_cls.h"

using namespace std;
using namespace cv;



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
    printf("Open model file [%s]\n", filename);
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
         attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

static int rknn_GetTop(float* pfProb, float* pfMaxProb, uint32_t* pMaxClass, uint32_t outputCount, uint32_t topNum)
{
  uint32_t i, j;

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


int rknn_cls::init(const char* model_path)
{
    int ret = 0;
    const int MODEL_IN_CHANNELS = 3;
    int model_len = 0;
    unsigned char * model = load_model(model_path, &model_len);
    ret   = rknn_init(&ctx, model, model_len, 1, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }
    free(model);
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }

    return 0;
}


int rknn_cls::detect(cv::Mat &orig_img,vector<clsInfo> &info)
{
    int ret = 0;
    if (!orig_img.data) {
        printf("cv::imread %s fail!\n");
        return -1;
    }
    cv::Mat orig_img_rgb;
    cv::cvtColor(orig_img, orig_img_rgb, cv::COLOR_BGR2RGB);

    if (orig_img.cols != MODEL_IN_WIDTH || orig_img.rows != MODEL_IN_HEIGHT) {
        // printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
        // cv::resize(orig_img, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), 0, 0, cv::INTER_LINEAR);
        cv::resize(orig_img_rgb, orig_img_rgb, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT));
    }
    // Set Input Data
    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type  = RKNN_TENSOR_UINT8;
    inputs[0].size  = orig_img_rgb.cols * orig_img_rgb.rows * orig_img_rgb.channels() * sizeof(uint8_t);
    inputs[0].fmt   = RKNN_TENSOR_NHWC;
    inputs[0].buf   = orig_img_rgb.data;

    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
    if (ret < 0) {
        printf("rknn_input_set fail! ret=%d\n", ret);
        return -1;
    }

    // Run
    // printf("rknn_run\n");
    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        printf("rknn_run fail! ret=%d\n", ret);
        return -1;
    }
    // Get Output
    rknn_output outputs[1];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    ret                   = rknn_outputs_get(ctx, 1, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return -1;
    }
    // Post Process
    for (int i = 0; i < io_num.n_output; i++) {
        uint32_t MaxClass[5];
        float    fMaxProb[5];
        float*   buffer = (float*)outputs[i].buf;
        uint32_t sz     = outputs[i].size / 4;

        // printf("%3d: %3d: \n", outputs[i].size,len);

        rknn_GetTop(buffer, fMaxProb, MaxClass, sz, 5);

        // printf(" --- Top1 ---\n");
        // printf("%3d: %8.6f\n", MaxClass[0], fMaxProb[0]);   //返回
        clsInfo ans;
        ans.lableIndex = MaxClass[0];
        ans.score = fMaxProb[0];
        info.push_back(ans);
    }
    

    // Release rknn_outputs
    rknn_outputs_release(ctx, 1, outputs);
    return 0;
}

int rknn_cls::release(){
    // Release
    if (ctx > 0)
    {
        rknn_destroy(ctx);
    }
    
    return 0;
}

rknn_cls::rknn_cls()
{
    ctx = 0;
    // MODEL_IN_WIDTH    = 224;
    // MODEL_IN_HEIGHT   = 224;
}

rknn_cls::~rknn_cls()
{
    int ret = release();
    if(!ret)
    {
        cout<<"[ERROR] rkCLs release fail\n";
    }
}

