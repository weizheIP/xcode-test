#if 0
#ifndef _CAR_THREAD
#define _CAR_THREAD



#include "post/model_base.h"


//extern cv::Mat get_split_merge(cv::Mat &img);


struct message_img
{
    string videoIndex;
    cv::Mat frame;
};

struct message_det
{
    string videoIndex;
    cv::Mat frame;
    vector<BoxInfo> detectData;
};

struct message_lpr_img
{
    string videoIndex;
    cv::Mat frame;
    BoxInfo inBox;
};

struct message_lpr
{
    string videoIndex;
    cv::Mat frame;
    BoxInfo detectData;
    string lprCode;
};

BlockingQueue<message_img> queue_pre;
BlockingQueue<message_lpr_img> queue_lpr_pre;
std::map<std::string, std::shared_ptr<BlockingQueue<message_det>>> queue_post;
std::map<std::string, std::shared_ptr<BlockingQueue<message_lpr>>> queue_lpr_post;
std::mutex queue_post_lock;
std::mutex queue_lpr_post_lock;

static void MallocCarPost(std::string rtsp_path)
{
	queue_post_lock.lock();
	queue_post[rtsp_path] = std::make_shared<BlockingQueue<message_det>>(10);
	queue_post_lock.unlock();
	queue_lpr_post_lock.lock();
	queue_lpr_post[rtsp_path] = std::make_shared<BlockingQueue<message_lpr>>(10);
	queue_lpr_post_lock.unlock();
}

static void DeleteCarPost(std::string rtsp_path)
{
	queue_post_lock.lock();
	queue_post.erase(rtsp_path);
	queue_post_lock.unlock();
	queue_lpr_post_lock.lock();
	queue_lpr_post.erase(rtsp_path);
	queue_lpr_post_lock.unlock();
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


class CarInfer
{
public:
    bool APP_Status = true;
    std::unique_ptr<std::thread> thread_main;
    std::unique_ptr<std::thread> thread_main2;
	std::string alg_name;

	void init(std::string rtsp_path_, std::string alg_name_) {
		alg_name = alg_name_;
//		thread_main = std::make_unique<std::thread>(&CarInfer::run, this);
        // infer.init("m_enc.pt",1,1); //只用npu核1，多次载入导致pt_set_core_mask报错
        
#ifdef RK
			infer.init(alg_name + "/m_enc.pt", 1, 3); // 核3
#else
			infer.init(alg_name + "/m_enc.pt"); // 核3
#endif

//		thread_main2 = std::make_unique<std::thread>(&CarInfer::run2, this);

        plateDetPtr_ = new lmYOLO();
        plateRecPtr_ = new recCarPlate();

#ifdef RK
	        plateDetPtr_->init(alg_name + "/det.rknn", 0);
	        plateRecPtr_->init(alg_name + "/rec.rknn");
#else
			plateDetPtr_->init(alg_name + "/det.pt", 0);
			plateRecPtr_->init(alg_name + "/rec.pt");
#endif
		b_init = true;
	}


    void run()
    {
        message_img premsg;

        detYOLO infer;
        // infer.init("m_enc.pt",1,1); //只用npu核1，多次载入导致pt_set_core_mask报错
        
#ifdef RK
			infer.init(alg_name + "/m_enc.pt", 1, 3); // 核3
#else
			infer.init(alg_name + "/m_enc.pt"); // 核3
#endif
        while (APP_Status)
        {
            if (queue_pre.IsEmpty())
            {
                usleep(1000);
                continue;
            }   
            if (queue_pre.Pop(premsg) != 0)
            {
                usleep(1000);
                continue;
            }
            message_det msg;
            vector<BoxInfo> carBoxInfoVec;
            carBoxInfoVec = infer.infer(premsg.frame);
            msg.videoIndex = premsg.videoIndex;
            msg.frame = premsg.frame;
            msg.detectData = carBoxInfoVec;
            queue_post_lock.lock();
            std::shared_ptr<BlockingQueue<message_det>> aa = queue_post[premsg.videoIndex];
            queue_post_lock.unlock();
            if (aa->Push(msg) != 0)
            {
                usleep(1000);
            }
        }
    }
    void run2()
    {
        message_lpr_img premsg;
        message_lpr msg;

        lmYOLO *plateDetPtr_;
        recCarPlate *plateRecPtr_;

        plateDetPtr_ = new lmYOLO();
        plateRecPtr_ = new recCarPlate();

		#ifdef RK
	        plateDetPtr_->init(alg_name + "/det.rknn", 0);
	        plateRecPtr_->init(alg_name + "/rec.rknn");
		#else
			plateDetPtr_->init(alg_name + "/det.pt", 0);
			plateRecPtr_->init(alg_name + "/rec.pt");
		#endif
		
        while (APP_Status)
        {
            if (queue_lpr_pre.IsEmpty())
            {
                usleep(1000);
                continue;
            } 
            if (queue_lpr_pre.Pop(premsg) != 0)
            {
                usleep(1000);
                continue;
            }
            cv::Mat img = premsg.frame; // 大图
            BoxInfo box = premsg.inBox; // 车框
            string code = "";
            BoxInfo carCodeBox;

            cv::Mat carImg = img(cv::Rect(box.x1, box.y1, (box.x2 - box.x1), (box.y2 - box.y1)));

            std::vector<CarBoxInfo> carPlateBoxes = plateDetPtr_->infer(carImg);
            int index;

            index = chooseBestCarPlateIndex(carPlateBoxes, box.y2 - box.y1);

            if (index != -1)
            {

                if (carPlateBoxes.size())
                {
                    int xmin = int(carPlateBoxes[index].x1);
                    int ymin = int(carPlateBoxes[index].y1);
                    int xmax = int(carPlateBoxes[index].x2);
                    int ymax = int(carPlateBoxes[index].y2);

                    cv::Point2f points_ref[] = {
                        cv::Point2f(0, 0),
                        cv::Point2f(100, 0),
                        cv::Point2f(100, 32),
                        cv::Point2f(0, 32)};

                    cv::Point2f points[] = {
                        cv::Point2f(float(carPlateBoxes[index].landmark[0] - xmin), float(carPlateBoxes[index].landmark[1] - ymin)),
                        cv::Point2f(float(carPlateBoxes[index].landmark[2] - xmin), float(carPlateBoxes[index].landmark[3] - ymin)),
                        cv::Point2f(float(carPlateBoxes[index].landmark[4] - xmin), float(carPlateBoxes[index].landmark[5] - ymin)),
                        cv::Point2f(float(carPlateBoxes[index].landmark[6] - xmin), float(carPlateBoxes[index].landmark[7] - ymin))};
                    cv::Mat M = cv::getPerspectiveTransform(points, points_ref);
                    cv::Mat img_box = carImg(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));

                    cv::Mat processed; // 掰直后的车牌img
                    // cv::Mat processed2; //掰直后的车牌img
                    cv::warpPerspective(img_box, processed, M, cv::Size(100, 32));
                    carCodeBox.x1 = carPlateBoxes[index].x1 + box.x1;
                    carCodeBox.x2 = carPlateBoxes[index].x2 + box.x1;
                    carCodeBox.y1 = carPlateBoxes[index].y1 + box.y1;
                    carCodeBox.y2 = carPlateBoxes[index].y2 + box.y1;
                    carCodeBox.label = carPlateBoxes[index].label;
                    carCodeBox.score = carPlateBoxes[index].score;

                    if (carPlateBoxes[index].label == 1)
                    {
                        processed = get_split_merge(processed);
                    }

                    // cv::imwrite("save1.jpg",processed);

                    // #ifdef __ONLYONE
                    // code = carRecOnlyOne_->rec(processed);
                    // #else
                    code = plateRecPtr_->infer(processed);
                }
            }

            msg.videoIndex = premsg.videoIndex;
            msg.frame = premsg.frame;
            msg.detectData = carCodeBox;
            msg.lprCode = check_chinese_car_plate_(code);
            
            queue_lpr_post_lock.lock();
            std::shared_ptr<BlockingQueue<message_lpr>> aa = queue_lpr_post[premsg.videoIndex];
            queue_lpr_post_lock.unlock();
            if (aa->Push(msg) != 0)
            {
                usleep(1000);
            }
        }
		delete(plateDetPtr_);		
		delete(plateRecPtr_);
    }
	
    void run2(message_lpr_img premsg_, message_lpr msg_)
    {
        message_lpr_img premsg = premsg_;
        message_lpr msg;

		
        if (b_init)
        {
            cv::Mat img = premsg.frame; // 大图
            BoxInfo box = premsg.inBox; // 车框
            string code = "";
            BoxInfo carCodeBox;

            cv::Mat carImg = img(cv::Rect(box.x1, box.y1, (box.x2 - box.x1), (box.y2 - box.y1)));

            std::vector<CarBoxInfo> carPlateBoxes = plateDetPtr_->infer(carImg);
            int index;

            index = chooseBestCarPlateIndex(carPlateBoxes, box.y2 - box.y1);

            if (index != -1)
            {

                if (carPlateBoxes.size())
                {
                    int xmin = int(carPlateBoxes[index].x1);
                    int ymin = int(carPlateBoxes[index].y1);
                    int xmax = int(carPlateBoxes[index].x2);
                    int ymax = int(carPlateBoxes[index].y2);

                    cv::Point2f points_ref[] = {
                        cv::Point2f(0, 0),
                        cv::Point2f(100, 0),
                        cv::Point2f(100, 32),
                        cv::Point2f(0, 32)};

                    cv::Point2f points[] = {
                        cv::Point2f(float(carPlateBoxes[index].landmark[0] - xmin), float(carPlateBoxes[index].landmark[1] - ymin)),
                        cv::Point2f(float(carPlateBoxes[index].landmark[2] - xmin), float(carPlateBoxes[index].landmark[3] - ymin)),
                        cv::Point2f(float(carPlateBoxes[index].landmark[4] - xmin), float(carPlateBoxes[index].landmark[5] - ymin)),
                        cv::Point2f(float(carPlateBoxes[index].landmark[6] - xmin), float(carPlateBoxes[index].landmark[7] - ymin))};
                    cv::Mat M = cv::getPerspectiveTransform(points, points_ref);
                    cv::Mat img_box = carImg(cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin));

                    cv::Mat processed; // 掰直后的车牌img
                    // cv::Mat processed2; //掰直后的车牌img
                    cv::warpPerspective(img_box, processed, M, cv::Size(100, 32));
                    carCodeBox.x1 = carPlateBoxes[index].x1 + box.x1;
                    carCodeBox.x2 = carPlateBoxes[index].x2 + box.x1;
                    carCodeBox.y1 = carPlateBoxes[index].y1 + box.y1;
                    carCodeBox.y2 = carPlateBoxes[index].y2 + box.y1;
                    carCodeBox.label = carPlateBoxes[index].label;
                    carCodeBox.score = carPlateBoxes[index].score;

                    if (carPlateBoxes[index].label == 1)
                    {
                        processed = get_split_merge(processed);
                    }

                    // cv::imwrite("save1.jpg",processed);

                    // #ifdef __ONLYONE
                    // code = carRecOnlyOne_->rec(processed);
                    // #else
                    code = plateRecPtr_->infer(processed);
                }
            }

            msg.videoIndex = premsg.videoIndex;
            msg.frame = premsg.frame;
            msg.detectData = carCodeBox;
            msg.lprCode = check_chinese_car_plate_(code);

			msg_ = msg;
            
        }
    }
	void stop() {
		APP_Status = false;
		if (thread_main && thread_main->joinable())
			thread_main->join();
		thread_main.reset();
	
		if (thread_main2 && thread_main2->joinable())
			thread_main2->join();
		thread_main2.reset();
	}

    ~CarInfer()
    {
    	if (b_init){
			delete(plateDetPtr_);		
			delete(plateRecPtr_);
		}
        stop();
    }

private:
	/* 车识别 */
	detYOLO infer;

	/* 车牌识别 */
	lmYOLO *plateDetPtr_;
	recCarPlate *plateRecPtr_;
	
	bool b_init = false;
};




#endif

#endif
