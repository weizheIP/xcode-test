#include "car_det.h"
#include "common/license/license.h"

static std::vector<float> LetterboxImage(const cv::Mat &src, cv::Mat &dst, const cv::Size &out_size)
{
    auto in_h = static_cast<float>(src.rows);
    auto in_w = static_cast<float>(src.cols);
    float out_h = out_size.height;
    float out_w = out_size.width;

    float scale = std::min(out_w / in_w, out_h / in_h);

    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);

    cv::resize(src, dst, cv::Size(mid_w, mid_h));

    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
    int left = (static_cast<int>(out_w) - mid_w) / 2;
    int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
    return pad_info;
}

YOLOV7_car::YOLOV7_car(Net_config_1 config)
{
	string model_path = config.modelpath;

	unsigned char *outdata;
    FILE *f = fopen(model_path.c_str(),"r");
    fseek(f,0,SEEK_END);
    size_t size = ftell(f);
    fseek(f,0,SEEK_SET);
    unsigned char* buff = (unsigned char*)malloc(size);
    fread(buff,size,1,f);
    fclose(f);        
    get_model_decrypt_value(buff,outdata,size);

	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;

	
	// std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
	OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	// ort_session = new Session(env, model_path.c_str(), sessionOptions);
	// // ort_session = new Ort::Session(env, static_cast<const void*>(buff),sizeof(buff), sessionOptions); 
	ort_session = new Ort::Session(env, buff,size, sessionOptions); 
	free(buff);

	size_t numInputNodes = ort_session->GetInputCount();
	size_t numOutputNodes = ort_session->GetOutputCount();
	AllocatorWithDefaultOptions allocator;
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
		auto input_dims = input_tensor_info.GetShape();
		input_node_dims.push_back(input_dims);
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];
	this->inpWidth = input_node_dims[0][3];

}

void YOLOV7_car::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < row; i++)
		{
			for (int j = 0; j < col; j++)
			{
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
				this->input_image_[c * row * col + i * col + j] = pix / 255.0;
			}
		}
	}
}

void YOLOV7_car::nms(vector<Yolov5Face_BoxStruct>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](Yolov5Face_BoxStruct a, Yolov5Face_BoxStruct b) { return a.score > b.score; });
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}

	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < int(input_boxes.size()); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < int(input_boxes.size()); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

			float w = (max)(float(0), xx2 - xx1 + 1);
			float h = (max)(float(0), yy2 - yy1 + 1);
			float inter = w * h;
			float ovr = inter / (vArea[i] + vArea[j] - inter);

			if (ovr >= this->nmsThreshold)
			{
				isSuppressed[j] = true;
			}
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const Yolov5Face_BoxStruct& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

vector<Yolov5Face_BoxStruct> YOLOV7_car::detect(Mat& frame)
{
	Mat dstimg;
	std::vector<float> pad_info = LetterboxImage(frame, dstimg, cv::Size(this->inpWidth, this->inpWidth));

	// resize(frame, dstimg, Size(this->inpWidth, this->inpHeight));
	cv::cvtColor(dstimg, dstimg, cv::COLOR_BGR2RGB);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	vector<Yolov5Face_BoxStruct> generate_boxes;
	
	Ort::Value &predictions = ort_outputs.at(0);
	auto pred_dims = predictions.GetTensorTypeAndShapeInfo().GetShape();
	num_proposal = pred_dims.at(1);
	nout = pred_dims.at(2);
	// [1,100800,14]

	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, k = 0; ///cx,cy,w,h,box_score, , x1,y1,score1, ...., x4,y4,score5,class_score
	const float* pdata = predictions.GetTensorMutableData<float>();
	for (n = 0; n < this->num_proposal; n++)   ///特征图尺度
	{
		float box_score = pdata[4];
		if (box_score > this->confThreshold)
		{
			float class_socre = box_score * pdata[13];
			if (class_socre > this->confThreshold)
			{
				float cx = pdata[0] ;  ///cx
				float cy = pdata[1] ;   ///cy
				float w = pdata[2] ;   ///w
				float h = pdata[3] ;  ///h

				float xmin = cx - 0.5 * w;
				float ymin = cy - 0.5 * h;
				float xmax = cx + 0.5 * w;
				float ymax = cy + 0.5 * h;
				xmin = (xmin - pad_info[0]) / pad_info[2];
				ymin = (ymin - pad_info[1]) / pad_info[2];
				xmax = (xmax - pad_info[0]) / pad_info[2];
				ymax = (ymax - pad_info[1]) / pad_info[2];

				k = 0;
				int x = int((pdata[5 + k]- pad_info[0]) / pad_info[2]);
				int y = int((pdata[5 + k + 1]- pad_info[1]) / pad_info[2]);
				Point kpt1 =  Point(x,y);
				k += 2;

				x = int((pdata[5 + k]- pad_info[0]) / pad_info[2]);
				y = int((pdata[5 + k + 1] - pad_info[1]) / pad_info[2]);
				Point kpt2 =  Point(x,y);
				k += 2;

				x = int((pdata[5 + k]- pad_info[0]) / pad_info[2]);
				y = int((pdata[5 + k + 1] - pad_info[1]) / pad_info[2]);
				Point kpt3 =  Point(x,y);
				k += 2;

				x = int((pdata[5 + k] - pad_info[0]) / pad_info[2]);
				y = int((pdata[5 + k + 1] - pad_info[1]) / pad_info[2]);
				Point kpt4 =  Point(x,y);
				
				
				xmin = std::max(std::min(xmin, (float)(frame.cols)), 0.f);
				ymin = std::max(std::min(ymin, (float)(frame.rows)), 0.f);
				xmax = std::max(std::min(xmax, (float)(frame.cols)), 0.f);
				ymax = std::max(std::min(ymax, (float)(frame.rows)), 0.f);
				std::vector<cv::Point> kkk={kpt1,kpt2,kpt3,kpt4};
				generate_boxes.push_back(Yolov5Face_BoxStruct{ static_cast<int>(xmin), static_cast<int>(ymin), static_cast<int>(xmax), static_cast<int>(ymax), class_socre,kkk  });
			}
		}
		pdata += nout;
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes);
	return generate_boxes;
}

vector<Yolov5Face_BoxStruct> YOLOV7_car::detect_v8(Mat& frame)
{
	Mat dstimg;
	std::vector<float> pad_info = LetterboxImage(frame, dstimg, cv::Size(this->inpWidth, this->inpWidth));

	// resize(frame, dstimg, Size(this->inpWidth, this->inpHeight));
	cv::cvtColor(dstimg, dstimg, cv::COLOR_BGR2RGB);
	this->normalize_(dstimg);
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	vector<Yolov5Face_BoxStruct> generate_boxes;
	
	Ort::Value &predictions = ort_outputs.at(0);
	auto pred_dims = predictions.GetTensorTypeAndShapeInfo().GetShape();
	num_proposal = pred_dims.at(2);
	nout = pred_dims.at(1);
	// [1,13,33600]
	auto pdata1 = predictions.GetTensorMutableData<float>();
	cv::Mat rawData = cv::Mat(nout, num_proposal, CV_32F, pdata1).t(); // [1,33600,13]
	float* pdata = (float*)rawData.data;

	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	int n = 0, k = 0; ///cx,cy,w,h,class_score, x1,y1,score1, ...., x4,y4,score5
	// const float* pdata = predictions.GetTensorMutableData<float>();
	for (n = 0; n < this->num_proposal; n++)   ///特征图尺度
	{
		float box_score = pdata[4];
		if (box_score > this->confThreshold)
		{			
				float cx = pdata[0] ;  ///cx
				float cy = pdata[1] ;   ///cy
				float w = pdata[2] ;   ///w
				float h = pdata[3] ;  ///h

				float xmin = cx - 0.5 * w;
				float ymin = cy - 0.5 * h;
				float xmax = cx + 0.5 * w;
				float ymax = cy + 0.5 * h;
				xmin = (xmin - pad_info[0]) / pad_info[2];
				ymin = (ymin - pad_info[1]) / pad_info[2];
				xmax = (xmax - pad_info[0]) / pad_info[2];
				ymax = (ymax - pad_info[1]) / pad_info[2];

				k = 0;
				int x = int((pdata[5 + k]- pad_info[0]) / pad_info[2]);
				int y = int((pdata[5 + k + 1]- pad_info[1]) / pad_info[2]);
				Point kpt1 =  Point(x,y);
				k += 2;

				x = int((pdata[5 + k]- pad_info[0]) / pad_info[2]);
				y = int((pdata[5 + k + 1] - pad_info[1]) / pad_info[2]);
				Point kpt2 =  Point(x,y);
				k += 2;

				x = int((pdata[5 + k]- pad_info[0]) / pad_info[2]);
				y = int((pdata[5 + k + 1] - pad_info[1]) / pad_info[2]);
				Point kpt3 =  Point(x,y);
				k += 2;

				x = int((pdata[5 + k] - pad_info[0]) / pad_info[2]);
				y = int((pdata[5 + k + 1] - pad_info[1]) / pad_info[2]);
				Point kpt4 =  Point(x,y);
				
				
				xmin = std::max(std::min(xmin, (float)(frame.cols)), 0.f);
				ymin = std::max(std::min(ymin, (float)(frame.rows)), 0.f);
				xmax = std::max(std::min(xmax, (float)(frame.cols)), 0.f);
				ymax = std::max(std::min(ymax, (float)(frame.rows)), 0.f);
				std::vector<cv::Point> kkk={kpt1,kpt2,kpt3,kpt4};
				generate_boxes.push_back(Yolov5Face_BoxStruct{ static_cast<int>(xmin), static_cast<int>(ymin), static_cast<int>(xmax), static_cast<int>(ymax), box_score,kkk  });
		}
		pdata += nout;
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes);
	return generate_boxes;
}