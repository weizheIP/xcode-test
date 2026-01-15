

#include "infer/rk_seg/rk_seg.h"

#define PERF_WITH_POST 1

using namespace std;
using namespace cv;

//----
// 找到最大的轮廓并返回
static std::vector<cv::Point> findBiggestContour(cv::Mat img) {
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  // cv::imwrite("save1.jpg",img);

  // 找到所有轮廓
  cv::findContours(img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  

  // 如果没有找到轮廓则返回
  if(contours.empty()) return std::vector<Point>();

  // 找最大的轮廓
  int largestContourIdx = 0;
  int largestContourSize = 0;
  for(int i = 0; i < contours.size(); i++) {
    if(contours[i].size() > largestContourSize) {
      largestContourSize = contours[i].size();
      largestContourIdx = i; 
    }
  }

  // 返回最大的轮廓
  
  // cv::drawContours(img, contours, -1, cv::Scalar(0, 0, 255), 2);

  // cv::imwrite("save2.jpg",img);

  return contours[largestContourIdx];
}


inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }



static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds, std::vector<int>& order,
               int filterId, float threshold)
{
  for (int i = 0; i < validCount; ++i) {
    if (order[i] == -1 || classIds[i] != filterId) {
      continue;
    }
    int n = order[i];
    for (int j = i + 1; j < validCount; ++j) {
      int m = order[j];
      if (m == -1 || classIds[i] != filterId) {
        continue;
      }
      float xmin0 = outputLocations[n * 4 + 0];
      float ymin0 = outputLocations[n * 4 + 1];
      float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
      float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];

      float xmin1 = outputLocations[m * 4 + 0];
      float ymin1 = outputLocations[m * 4 + 1];
      float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
      float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];

      float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

      if (iou > threshold) {
        order[j] = -1;
      }
    }
  }
  return 0;
}

static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
{
  float key;
  int   key_index;
  int   low  = left;
  int   high = right;
  if (left < right) {
    key_index = indices[left];
    key       = input[left];
    while (low < high) {
      while (low < high && input[high] <= key) {
        high--;
      }
      input[low]   = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key) {
        low++;
      }
      input[high]   = input[low];
      indices[high] = indices[low];
    }
    input[low]   = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
  }
  return low;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
  float  dst_val = (f32 / scale) + zp;
  int8_t res     = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

static int process(float* input, int* anchor, int grid_h, int grid_w, int height, int width, int stride,
                   std::vector<float>& boxes, std::vector<std::vector<float>>& mask_feat, std::vector<float>& objProbs, std::vector<int>& classId, float threshold,
                   int32_t zp, float scale)
{
  // int OBJ_CLASS_NUM = 1; // 114/3 -32-5 
  int PROP_BOX_SIZE =5+OBJ_CLASS_NUM+32;

  int    validCount = 0;
  int    grid_len   = grid_h * grid_w;
  float  thres      = unsigmoid(threshold);
  // int8_t thres_i8   = qnt_f32_to_affine(thres, zp, scale);
  for (int a = 0; a < 3; a++) {
    for (int i = 0; i < grid_h; i++) {
      for (int j = 0; j < grid_w; j++) {
        float box_confidence = input[(PROP_BOX_SIZE * a + 4) * grid_len + i * grid_w + j];
        if (box_confidence >= thres) {
          int     offset = (PROP_BOX_SIZE * a) * grid_len + i * grid_w + j;
          float* in_ptr = input + offset;          
          float   box_x  = sigmoid(*in_ptr) * 2.0 - 0.5;
          float   box_y  = sigmoid(in_ptr[grid_len]) * 2.0 - 0.5;
          float   box_w  = sigmoid(in_ptr[2 * grid_len]) * 2.0;
          float   box_h  = sigmoid(in_ptr[3 * grid_len]) * 2.0;
          box_x          = (box_x + j) * (float)stride;
          box_y          = (box_y + i) * (float)stride;
          box_w          = box_w * box_w * (float)anchor[a * 2];
          box_h          = box_h * box_h * (float)anchor[a * 2 + 1];
          box_x -= (box_w / 2.0);
          box_y -= (box_h / 2.0);

          float maxClassProbs = in_ptr[5 * grid_len];
          int    maxClassId    = 0;
          for (int k = 1; k < OBJ_CLASS_NUM; ++k) {
            float prob = in_ptr[(5 + k) * grid_len];
            if (prob > maxClassProbs) {
              maxClassId    = k;
              maxClassProbs = prob;
            }
          }
          if (maxClassProbs>thres){
            // objProbs.push_back(sigmoid(deqnt_affine_to_f32(maxClassProbs, zp, scale))* sigmoid(deqnt_affine_to_f32(box_confidence, zp, scale)));
            objProbs.push_back(sigmoid(maxClassProbs)* sigmoid(box_confidence));
            classId.push_back(maxClassId);
            validCount++;
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
            // std::vector<float> temp_proto(in_ptr + 5 + num_class, in_ptr + 5 + num_class + 32);
            std::vector<float> temp_proto;
            for (int tk = 0; tk < 32; ++tk) {
              temp_proto.push_back(in_ptr[(6 + tk) * grid_len]);
            }
            mask_feat.push_back(temp_proto);
          }
        }
      }
    }
  }
  // std::cout <<"validCount----"<< validCount<<"--"<< std::endl;
  return validCount;
}

// static int post_process(float* input0, float* input1, float* input2, float* maskData, int model_in_h, int model_in_w, float conf_threshold,
static int post_process(float* input0, float* input1, float* input2,float* input3, float* maskData, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales, detect_result_group_t2* group)
{
  static int init = -1;
  if (init == -1) {
    int ret = 0;
    // ret     = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    // if (ret < 0) {
    //   return -1;
    // }

    init = 0;
  }
  // std::cout <<"post_process----"<< 1<<"--"<< std::endl;

  memset(group, 0, sizeof(detect_result_group_t2));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int>   classId;
  std::vector<std::vector<float>> mask_feat;

  // stride 8
  int stride0     = 8;
  int grid_h0     = model_in_h / stride0;
  int grid_w0     = model_in_w / stride0;
  int validCount0 = 0;
  validCount0 = process(input0, (int*)anchor0, grid_h0, grid_w0, model_in_h, model_in_w, stride0, filterBoxes,mask_feat, objProbs,
                        classId, conf_threshold, qnt_zps[0], qnt_scales[0]);

  // stride 16
  int stride1     = 16;
  int grid_h1     = model_in_h / stride1;
  int grid_w1     = model_in_w / stride1;
  int validCount1 = 0;
  validCount1 = process(input1, (int*)anchor1, grid_h1, grid_w1, model_in_h, model_in_w, stride1, filterBoxes,mask_feat, objProbs,
                        classId, conf_threshold, qnt_zps[1], qnt_scales[1]);

  // stride 32
  int stride2     = 32;
  int grid_h2     = model_in_h / stride2;
  int grid_w2     = model_in_w / stride2;
  int validCount2 = 0;
  validCount2 = process(input2, (int*)anchor2, grid_h2, grid_w2, model_in_h, model_in_w, stride2, filterBoxes,mask_feat, objProbs,
                        classId, conf_threshold, qnt_zps[2], qnt_scales[2]);

  // stride 64
  int stride3 = 64;
  int grid_h3 = model_in_h / stride3;
  int grid_w3 = model_in_w / stride3;
  int validCount3 = 0;
  validCount3 = process(input3, (int *)anchor3, grid_h3, grid_w3, model_in_h, model_in_w, stride3, filterBoxes,mask_feat, objProbs, 
                        classId, conf_threshold, qnt_zps[3], qnt_scales[3]);


  int validCount = validCount0 + validCount1 + validCount2 + validCount3;
  // int validCount = validCount0 + validCount1 + validCount2;
  // no object detect
  if (validCount <= 0) {
    return 0;
  }

  std::vector<int> indexArray;
  for (int i = 0; i < validCount; ++i) {
    indexArray.push_back(i);
  }

  quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

  std::set<int> class_set(std::begin(classId), std::end(classId));

  for (auto c : class_set) {
    nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
  }

  std::vector<std::vector<float>> temp_mask_proposals;
  std::vector<cv::Rect> rect_160s;

  int last_count = 0;
  group->count   = 0;
  /* box valid detect target */
  for (int i = 0; i < validCount; ++i) {
    if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) {
      continue;
    }
    int n = indexArray[i];

    float x1       = filterBoxes[n * 4 + 0];
    float y1       = filterBoxes[n * 4 + 1];
    float x2       = x1 + filterBoxes[n * 4 + 2];
    float y2       = y1 + filterBoxes[n * 4 + 3];
    int   id       = classId[n];
    float obj_conf = objProbs[i];

    temp_mask_proposals.push_back(mask_feat[n]);

    if(x1<0)
      x1 = 0;
    if(x1 > model_in_w)
      x1 = model_in_w;
    if(x2<0)
      x2 = 0;
    if(x2 > model_in_w)
      x2 = model_in_w;


    if(y1<0)
      y1 = 0;
    if(y2<0)
      y2 = 0;
    
    if(y1 > model_in_h)
      y1 = model_in_h;

    if(y2 > model_in_h)
      y2 = model_in_h;
    
    
    // int nx = x1/4;
    // int ny = y1/4;
    // int nw = x2/4 - x1/4;
    // int nh = y2/4 - y1/4;
    // if((nx + nw) > model_in_w/4 )
    //   nw = model_in_w/4 - nx;
    // if((ny) > model_in_h/4)
    //   nh = model_in_h/4 - nh;

    // rect_160s.push_back(cv::Rect(int(nx), int(ny), int(nw), int(nh)));

    int x1_int = int(x1/4);
    int y1_int = int(y1/4);
    int w_int = int(x2/4 - x1/4);
    int h_int = int(y2/4 - y1/4);
    // if((w_int == 0) || (h_int == 0) )
    //   continue;
    rect_160s.push_back(cv::Rect(x1_int, y1_int, w_int, h_int));

    
    group->results[last_count].box.left   = (int)(clamp(x1, 0, model_in_w) );
    group->results[last_count].box.top    = (int)(clamp(y1, 0, model_in_h) );
    group->results[last_count].box.right  = (int)(clamp(x2, 0, model_in_w) );
    group->results[last_count].box.bottom = (int)(clamp(y2, 0, model_in_h) );
    group->results[last_count].prop       = obj_conf;
    group->results[last_count].name       = id;
    // char* label                           = id;
    // strncpy(group->results[last_count].name, label, OBJ_NAME_MAX_SIZE);

    // printf("result %2d: (%4d, %4d, %4d, %4d), %s\n", i, group->results[last_count].box.left,
    // group->results[last_count].box.top,
    //        group->results[last_count].box.right, group->results[last_count].box.bottom, label);
    last_count++;
  }
  group->count = last_count;
  // std::cout <<"num----"<< rect_160s.size()<<"--"<< std::endl;

// #ifndef _FISH_SEG
//   // 处理mask
//   #ifdef Model640
//   int modelsize_4=640/4;
//   #else
  int modelsize_4=1280/4;
  // #endif
	cv::Mat maskProposals;
	for (int i = 0; i < temp_mask_proposals.size(); ++i)
		maskProposals.push_back( cv::Mat(temp_mask_proposals[i]).t() );

	float* pdata = maskData;
	std::vector<float> mask(pdata, pdata + 32 * modelsize_4 * modelsize_4);
	
	cv::Mat mask_protos = cv::Mat(mask);
	// cv::Mat protos = mask_protos.reshape(0, { 32,160 * 160 });//将prob1的值 赋给mask_protos
	cv::Mat protos = mask_protos.reshape(0, 32);//将prob1的值 赋给mask_protos
	
	cv::Mat matmulRes = (maskProposals * protos).t();//n*32 32*25600 A*B是以数学运算中矩阵相乘的方式实现的，要求A的列数等于B的行数时 160 * 160=25600
	// cv::Mat masks = matmulRes.reshape(objects.size(), { 160,160 });
	cv::Mat masks = matmulRes.reshape(rect_160s.size(), modelsize_4);
	// std::cout << protos.size()<<"----"<< masks.size()<<masks.channels() << endl;
	std::vector<cv::Mat> maskChannels;
	cv::split(masks, maskChannels);
  // std::cout << "mask num: "<<maskChannels.size()<<std::endl;
  // cv::Rect holeImgRect(0, 0, img_w, img_h);
	
	for (int i = 0; i < rect_160s.size(); ++i) {
    
    if(rect_160s[i].width == 0 || rect_160s[i].height == 0 )
      continue;


		cv::Mat dest, mask, dest160 ;
		//sigmoid
		cv::exp(-maskChannels[i], dest);
		dest = 1.0 / (1.0 + dest);//160*160
    // std::cout <<"--------"<< dest<<endl;
    dest160=dest.clone();
		cv::Rect temp_rect160 = rect_160s[i] ;		
		cv::Mat mask160 = dest160(temp_rect160) > 0.5;        
    // cv::imwrite("out/"+std::to_string(i)+"outm.jpg",mask160*255);
    int nCount = cv::countNonZero(mask160);
    // std::cout << "nCount:" << nCount<<" size:"<<mask.size() << std::endl;
		// objects[i].boxMask = mask;
		// objects[i].area = nCount;
    group->results[i].area = nCount;
    // group->results[i].cc = findBiggestContour((dest160 > 0.5)*255);
    // group->results[i].cc =
    std::vector<cv::Point> tmp = findBiggestContour(mask160*255);
    for(int j =0 ;j< tmp.size(); j++)
    {
      group->results[i].cc.push_back(cv::Point(tmp[j].x + temp_rect160.x,tmp[j].y + temp_rect160.y));
    }
	}
// #endif

  return 0;
}

//---
// static void dump_tensor_attr(rknn_tensor_attr* attr)
// {
//   printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
//          "zp=%d, scale=%f\n",
//          attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
//          attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
//          get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
// }

double __get_us2(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

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
    printf("Open model file %s .\n", filename);
    fseek(fp, 0, SEEK_END);
    int size = ftell(fp);

    data = load_data(fp, 0, size);

    fclose(fp);

    *model_size = size;
    return data;
}

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

static cv::Mat resize_image(cv::Mat srcimg, int *newh, int *neww, int *top, int *left)
{
  // #ifdef Model640
  // int inpHeight=640;
  // int inpWidth=640;
  // #else
  int inpHeight=1280;
  int inpWidth=1280;
  // #endif
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
      // if((*neww) == inpWidth)
      if(0)
      {
        // printf("no resize needed\n");
        dstimg = srcimg;
      }
      else
      {
        // printf("resize %d,%d\n",*newh,*neww);
        cv::resize(srcimg, dstimg, cv::Size(*neww, *newh), cv::INTER_AREA);
      } 
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



rk_seg::rk_seg()
{
  // nms_threshold = NMS_THRESH;
  // box_conf_threshold = BOX_THRESH;
  status     = 0;
  actual_size        = 0;
  img_width          = 0;
  img_height         = 0;
  img_channel        = 0;
  output_attrs = NULL;
}

rk_seg::~rk_seg()
{
    if(output_attrs)
      delete output_attrs;
    ret = rknn_destroy(ctx);
    if(ret != 0)
        printf("[ERROR] rknn_Seg release fail!\n");
}

int rk_seg::init(string path)
{
    // printf("post process config: box_conf_threshold = %.2f, nms_threshold = %.2f\n", box_conf_threshold, nms_threshold);
    printf("Loading mode...\n");
    int            model_data_size = 0;
    unsigned char* model_data      = load_model(path.c_str(), &model_data_size);
    ret                            = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
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
    // printf("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);

    // rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        printf("rknn_init error ret=%d\n", ret);
        return -1;
    }
    // printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

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

    // rknn_tensor_attr output_attrs[io_num.n_output];
    output_attrs = new rknn_tensor_attr[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        // dump_tensor_attr(&(output_attrs[i]));
    }

    // int channel = 3;
    // int width   = 0;
    // int height  = 0;
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
    return 0;
}

int rk_seg::seg(cv::Mat &orig_img,std::vector<Object> &segs)
{
  timeval begintime;
  gettimeofday(&begintime, NULL);
  gettimeofday(&start_time, NULL);
  rknn_input inputs[1];
  memset(inputs, 0, sizeof(inputs));
  inputs[0].index        = 0;
  inputs[0].type         = RKNN_TENSOR_UINT8;
  inputs[0].size         = width * height * channel;
  inputs[0].fmt          = RKNN_TENSOR_NHWC;
  inputs[0].pass_through = 0;

  void* resize_buf = nullptr;

  cv::Mat img;
  cv::cvtColor(orig_img, img, cv::COLOR_BGR2RGB);
  img_width  = img.cols;
  img_height = img.rows;
  // printf("img width = %d, img height = %d\n", img_width, img_height);


  // 加letterbox
  int newh = 0, neww = 0, padh = 0, padw = 0;
  img = resize_image(img, &newh, &neww, &padh, &padw);
  inputs[0].buf = (void*)img.data;
  gettimeofday(&stop_time, NULL);
  // printf("perfer: %f ms\n", (__get_us2(stop_time) - __get_us2(begintime)) / 1000);
  gettimeofday(&start_time, NULL);
  rknn_inputs_set(ctx, io_num.n_input, inputs);

  rknn_output outputs[io_num.n_output];
  memset(outputs, 0, sizeof(outputs));
  for (int i = 0; i < io_num.n_output; i++) {
    // outputs[i].want_float = 0;
    outputs[i].want_float = 1;
  }

  ret = rknn_run(ctx, NULL);
  ret = rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);
  gettimeofday(&stop_time, NULL);
  // printf("only net: %f ms\n", (__get_us2(stop_time) - __get_us2(start_time)) / 1000);
  gettimeofday(&start_time, NULL);
  // post process
  float scale_w = (float)width / img_width;
  float scale_h = (float)height / img_height;

  detect_result_group_t2 detect_result_group;
  std::vector<float>    out_scales;
  std::vector<int32_t>  out_zps;
  for (int i = 0; i < io_num.n_output; ++i) {
    out_scales.push_back(output_attrs[i].scale);
    out_zps.push_back(output_attrs[i].zp);
  }

  post_process((float*)outputs[0].buf, (float*)outputs[1].buf, (float*)outputs[2].buf, (float*)outputs[3].buf, (float*)outputs[4].buf, height, width,
  // post_process((float*)outputs[0].buf, (float*)outputs[1].buf, (float*)outputs[2].buf, (float*)outputs[3].buf, height, width,
            box_conf_threshold, nms_threshold, scale_w, scale_h, out_zps, out_scales, &detect_result_group);

  // Draw Objects
  float ratioh = (float)orig_img.rows / newh, ratiow = (float)orig_img.cols / neww;
  char text[256];

  std::vector<std::vector<cv::Point>> pointVecs;

  for (int i = 0; i < detect_result_group.count; i++) {
    detect_result_t2* det_result = &(detect_result_group.results[i]);
    // sprintf(text, "%s %.1f%%", det_result->name, det_result->prop * 100);
    // sprintf(text, "%.1f", det_result->prop );
    // printf("%s @ (%d %d %d %d) %f\n", det_result->name, det_result->box.left, det_result->box.top,
    //        det_result->box.right, det_result->box.bottom, det_result->prop);
    // int x1 = det_result->box.left;
    // int y1 = det_result->box.top;
    // int x2 = det_result->box.right;
    // int y2 = det_result->box.bottom;

    int x1 = (det_result->box.left  -padw)* ratiow;
    int y1 = (det_result->box.top  -padh)* ratioh;
    int x2 = (det_result->box.right -padw)* ratiow;
    int y2 = (det_result->box.bottom  -padh)* ratioh;
    int area = det_result->area;
    sprintf(text, "%d %.1f", area, det_result->prop );
    struct Object info;
    info.rect.x = x1;
    info.rect.y = y1;
    info.rect.width = x2 -x1;
    info.rect.height = y2 - y1;
    info.area = area;
    info.prob = det_result->prop;
    for(int k = 0; k<det_result->cc.size(); k++)
    {
      info.cc.push_back(cv::Point((det_result->cc[k].x*4 -padw) * ratiow, (det_result->cc[k].y*4 -padh)* ratioh));
      // std::cout<<"["<<(det_result->cc[k].x*4 -padw) * ratiow <<"_"<<(det_result->cc[k].y*4 -padh)* ratioh<<"],";
    }
    // pointVecs.push_back(info.cc);

    // std::cout<<std::endl;

    segs.push_back(info);
    
    // printf("(%d %d %d %d %d %d %d %d %f %f) \n", x1,y1,x2,y2,newh,neww,padh,padw,ratioh,ratiow);
    // rectangle(orig_img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0, 255), 1);
    // putText(orig_img, text, cv::Point(int((x1+x2)/2), int((y1+y2)/2)), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
  }

  //测试可以把轮廓画出来
  // cv::drawContours(orig_img, pointVecs, -1, cv::Scalar(0, 0, 255), 2);

  // putText(orig_img, std::to_string(detect_result_group.count), cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 255));

  ret = rknn_outputs_release(ctx, io_num.n_output, outputs);

  

  gettimeofday(&stop_time, NULL);
  // printf("post net: %f ms\n", (__get_us2(stop_time) - __get_us2(start_time)) / 1000);
  // printf("all: %f ms\n", (__get_us2(stop_time) - __get_us2(begintime)) / 1000);

  return ret;

}



