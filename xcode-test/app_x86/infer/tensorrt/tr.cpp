#include "tr.h"
#include "common/license/license.h"


static const float CONF_THRESHOLD = 0.1;
static const float NMS_THRESHOLD = 0.6;
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output0";//detect

static Logger gLogger;
void TRtDet::doInference(IExecutionContext* context, float* input, float* output, int batchSize)
{
    // const ICudaEngine& engine = context->getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    // assert(_engine->getNbBindings() == 3);
    void* buffers[3];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = _engine->getBindingIndex(INPUT_BLOB_NAME);
	const int outputIndex = _engine->getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * _modelH * _modelW * sizeof(float)));//
	CHECK(cudaMalloc(&buffers[outputIndex], batchSize * _outPutSize * sizeof(float)));
	// cudaMalloc分配内存 cudaFree释放内存 cudaMemcpy或 cudaMemcpyAsync 在主机和设备之间传输数据
	// cudaMemcpy cudaMemcpyAsync 显式地阻塞传输 显式地非阻塞传输 
    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * _modelH * _modelW * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueue(batchSize, buffers, stream, nullptr);
    // context->enqueueV2(bindings, stream, nullptr);
	CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * _outPutSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}

int TRtDet::init(int gpu_device)
{
	cudaSetDevice(gpu_device); //选择gpu卡

    char* trtModelStream{ nullptr }; //char* trtModelStream==nullptr;  开辟空指针后 要和new配合使用，比如89行 trtModelStream = new char[size]
	size_t size{ 0 };//与int固定四个字节不同有所不同,size_t的取值range是目标平台下最大可能的数组尺寸,一些平台下size_t的范围小于int的正数范围,又或者大于unsigned int. 使用Int既有可能浪费，又有可能范围不够大。
    std::ifstream file(_modelPath, std::ios::binary);
    if (file.good()) {
		std::cout << "load engine success" << std::endl;
		file.seekg(0, file.end);//指向文件的最后地址
		size = file.tellg();//把文件长度告诉给size


		file.seekg(0, file.beg);//指回文件的开始地址
		trtModelStream = new char[size];//开辟一个char 长度是文件的长度
		assert(trtModelStream);//
		file.read(trtModelStream, size);//将文件内容传给trtModelStream
		file.close();//关闭
        
        unsigned char* modelvalue;
        get_model_decrypt_value((unsigned char*)trtModelStream, modelvalue, size);
        
	}
	else {
		std::cout << "[ERROR] load engine failed" << std::endl;
		return 1;
	}

    _runtime = createInferRuntime(gLogger);
	assert(_runtime != nullptr);
    // bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
    _engine = _runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(_engine != nullptr);
	_context = _engine->createExecutionContext();
    assert(_context != nullptr);
	delete[] trtModelStream;

    int inputBinding = _engine->getBindingIndex("images");
    nvinfer1::Dims inputDims = _engine->getBindingDimensions(inputBinding);
    _modelH = inputDims.d[2];
    _modelW = inputDims.d[3];

	int outputBinding = _engine->getBindingIndex("output0");
    nvinfer1::Dims outputDims = _engine->getBindingDimensions(outputBinding);
    _classNum = outputDims.d[1]-4;
    Num_box = outputDims.d[2];

	// _modelH = 1280;
	// _modelW = 1280;
	// _classNum = 1;

	_outPutSize = Num_box * (_classNum + 4);
	_prob = new float[_outPutSize];	
	_data = new float[3*_modelH*_modelW];

    // std::cout << "Input Resolution: " << _modelH << "x" << _modelW << std::endl;
    // std::cout << "Number of Classes: " << _classNum << std::endl;

	return 0;

}

TRtDet::~TRtDet()
{
    if(_context)
    {
        _context->destroy();
        _engine->destroy();
        _runtime->destroy();
        delete [] _prob;
		delete [] _data;
    }
}

int TRtDet::det(cv::Mat &src, std::vector<BoxInfo> &boxes2)
{

    if (src.empty())
        return -1;
    int img_width = src.cols;
	int img_height = src.rows;
	auto begin = std::chrono::system_clock::now();

    // float data[3 * _modelW * _modelH];
    cv::Mat pr_img0, pr_img;
    std::vector<int> padsize;
	pr_img = preprocess_img(src, _modelH, _modelW, padsize);       // Resize
	int newh = padsize[0], neww = padsize[1], padh = padsize[2], padw = padsize[3];
	float ratio_h = (float)src.rows / newh;
	float ratio_w = (float)src.cols / neww;
	int i = 0;// [1,3,INPUT_H,INPUT_W]
	//std::cout << "pr_img.step" << pr_img.step << std::endl;
	for (int row = 0; row < _modelH; ++row) {
		uchar* uc_pixel = pr_img.data + row * pr_img.step;//pr_img.step=widthx3 就是每一行有width个3通道的值
		for (int col = 0; col < _modelW; ++col)
		{
			// 已经是bgr转rgb了
			_data[i] = (float)uc_pixel[2] / 255.0;
			_data[i + _modelH * _modelW] = (float)uc_pixel[1] / 255.0;
			_data[i + 2 * _modelH * _modelW] = (float)uc_pixel[0] / 255.;
			uc_pixel += 3;
			++i;
		}
	}

    auto start = std::chrono::system_clock::now();
    doInference(_context, _data, _prob, 1);

    auto end = std::chrono::system_clock::now();
	// std::cout << "推理时间：" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 100 << "ms" << std::endl;

	std::vector<int> classIds;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
    
    int net_length = _classNum + 4 ;
    cv::Mat out1 = cv::Mat(net_length, Num_box, CV_32F, _prob);
    for (int i = 0; i < Num_box; i++) {
		//输出是1*net_length*Num_box;所以每个box的属性是每隔Num_box取一个值，共net_length个值
		cv::Mat scores = out1(cv::Rect(i, 4, 1, _classNum)).clone();
		cv::Point classIdPoint;
		double max_class_socre;
		cv::minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
		max_class_socre = (float)max_class_socre;
		if (max_class_socre >= CONF_THRESHOLD) {			
			float x = (out1.at<float>(0, i) - padw) * ratio_w;  //cx
			float y = (out1.at<float>(1, i) - padh) * ratio_h;  //cy
			float w = out1.at<float>(2, i) * ratio_w;  //w
			float h = out1.at<float>(3, i) * ratio_h;  //h
			int left = MAX((x - 0.5 * w), 0);
			int top = MAX((y - 0.5 * h), 0);
			int width = (int)w;
			int height = (int)h;
			if (width <= 1 || height <= 1 || max_class_socre>=1) { continue; } //多模型，出现很多score大于1的框
			classIds.push_back(classIdPoint.y);
			confidences.push_back(max_class_socre);
			boxes.push_back(cv::Rect(left, top, width, height));
		}
	}
    //执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	std::vector<int> nms_result;
	cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, nms_result);
	// std::vector<OutputSeg> output;
    
	cv::Rect holeImgRect(0, 0, src.cols, src.rows);
	for (int i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		// OutputSeg result;
		// result.id = classIds[idx];
		// result.confidence = confidences[idx];
		// result.box = boxes[idx]& holeImgRect;
        // output.push_back(result);

        BoxInfo box;
        box.x1 = boxes[idx].x;
		box.y1 = boxes[idx].y;
		box.x2 = boxes[idx].x + boxes[idx].width;
		box.y2 = boxes[idx].y + boxes[idx].height;
        box.score = confidences[idx];
        box.label = classIds[idx];
        boxes2.push_back(box);
		
	}
	// printf("boxes size: %d\n", boxes2.size());
	// auto final = std::chrono::system_clock::now();

    // std::cout << "后处理时间：" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    // DrawPred(src, output);
    // std::cout << "推理时间：" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 100 << "ms" << std::endl;
	
	// _useTimeArr.push_back((int)(std::chrono::duration_cast<std::chrono::milliseconds>(start - begin).count()));
	// _useTimeArr2.push_back((int)(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()));
	// _useTimeArr3.push_back((int)(std::chrono::duration_cast<std::chrono::milliseconds>(final - end).count()));
	
	// if(_useTimeArr3.size() % 100 == 0)
	// {
	// 	int preUse = std::accumulate(_useTimeArr.begin(), _useTimeArr.end(), 0) / _useTimeArr.size();
	// 	int detUse = std::accumulate(_useTimeArr2.begin(), _useTimeArr2.end(), 0) / _useTimeArr2.size();
	// 	int PostUse = std::accumulate(_useTimeArr3.begin(), _useTimeArr3.end(), 0) / _useTimeArr3.size();

	// 	std::cout<<"[DET_INFO] aver total:"<<preUse + detUse + PostUse<<",preUse:"<<preUse<<",detUse:"<<detUse<<",postUse:"<<PostUse<<", appName:"<<_appName<<"\n";
		
	// 	_useTimeArr.clear();
	// 	_useTimeArr2.clear();
	// 	_useTimeArr3.clear();
	// }

    return 0;
}






