#ifndef _CAR_THREAD
#define _CAR_THREAD


#include <thread>
#include <mutex>
#include <memory>
#include <atomic>
#include <string>
#include <vector>
#include <iostream>
#include <unistd.h>

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
    // ===== 单例入口 =====
    static CarInfer& instance() {
        std::call_once(init_flag_, [] {
            instance_.reset(new CarInfer());
        });
        return *instance_;
    }

    // ===== 生命周期控制 =====
    // 仅第一次调用会真正初始化模型与资源；后续重复调用安全但无副作用
    void init(const std::string& rtsp_path_, const std::string& alg_name_) {
        (void)rtsp_path_; // 你若需要可保存
        std::call_once(model_flag_, [&]{
            alg_name = alg_name_;
            // 初始化主检测器（成员，不要在 run() 里再创建/遮蔽）
#ifdef RK
            infer.init(alg_name + "/m_enc.pt", 1, 3); // 使用核3
#else
            infer.init(alg_name + "/m_enc.pt");
#endif
            // 初始化车牌检测/识别（成员）
            plateDetPtr_ = std::make_unique<lmYOLO>();
            plateRecPtr_ = std::make_unique<recCarPlate>();
#ifdef RK
            plateDetPtr_->init(alg_name + "/det.rknn", 0);
            plateRecPtr_->init(alg_name + "/rec.rknn");
#else
            plateDetPtr_->init(alg_name + "/det.pt", 0);
            plateRecPtr_->init(alg_name + "/rec.pt");
#endif
            b_init.store(true, std::memory_order_release);
        });
    }

    // 启动工作线程（可多次调用，只有第一次真正启动）
    void start() {
        bool expected = false;
        if (started_.compare_exchange_strong(expected, true)) {
            APP_Status.store(true, std::memory_order_release);
            thread_main  = std::make_unique<std::thread>(&CarInfer::run,  this);
            thread_main2 = std::make_unique<std::thread>(&CarInfer::run2, this);
        }
    }

    // 停止线程并回收
    void stop() {
        APP_Status.store(false, std::memory_order_release);
        if (thread_main && thread_main->joinable())  thread_main->join();
        if (thread_main2 && thread_main2->joinable()) thread_main2->join();
        thread_main.reset();
        thread_main2.reset();
        started_.store(false, std::memory_order_release);
    }

    // ===== 工作线程函数 =====
    void run()
    {
        message_img premsg;

        // 注意：不再创建/初始化局部 detYOLO，统一使用成员 infer
        while (APP_Status.load(std::memory_order_acquire))
        {
            if (queue_pre.IsEmpty()) { usleep(1000); continue; }
            if (queue_pre.Pop(premsg) != 0) { usleep(1000); continue; }

            message_det msg;
            std::vector<BoxInfo> carBoxInfoVec = infer.infer(premsg.frame);
            msg.videoIndex = premsg.videoIndex;
            msg.frame = premsg.frame;
            msg.detectData = carBoxInfoVec;

            queue_post_lock.lock();
            std::shared_ptr<BlockingQueue<message_det>> aa = queue_post[premsg.videoIndex];
            queue_post_lock.unlock();
            if (aa->Push(msg) != 0) { usleep(1000); }
        }
    }

    void run2()
    {
        // 注意：不再在此 new / init 局部 plateDetPtr_ / plateRecPtr_，统一使用成员
        message_lpr_img premsg;
        message_lpr msg;

        while (APP_Status.load(std::memory_order_acquire))
        {
            if (queue_lpr_pre.IsEmpty()) { usleep(1000); continue; }
            if (queue_lpr_pre.Pop(premsg) != 0) { usleep(1000); continue; }

            message_lpr out;
            if (plateInfer(premsg, out)) {
                queue_lpr_post_lock.lock();
                std::shared_ptr<BlockingQueue<message_lpr>> aa = queue_lpr_post[premsg.videoIndex];
                queue_lpr_post_lock.unlock();
                if (aa->Push(out) != 0) { usleep(1000); }
            }
        }
    }

    std::vector<BoxInfo> carInfer(cv::Mat& image)
    {
    	 std::vector<BoxInfo> carBoxInfoVec = {};
        if (!b_init.load(std::memory_order_acquire)) return carBoxInfoVec;
        {
			std::lock_guard<std::mutex> lk(car_infer_mu_);
	        carBoxInfoVec = infer.infer(image);
        }

		return carBoxInfoVec;
    }

    // 将原先 “void run2(message_lpr_img premsg_, message_lpr msg_)” 改为按引用输出；返回是否成功
    bool plateInfer(const message_lpr_img& premsg, message_lpr& msg_out)
    {
        if (!b_init.load(std::memory_order_acquire)) return false;

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

        msg_out.videoIndex = premsg.videoIndex;
        msg_out.frame = premsg.frame;
        msg_out.detectData = carCodeBox;
        msg_out.lprCode = check_chinese_car_plate_(code);
		std::cout << "code == " << code << std::endl;
        return true;
    }

    // 禁止拷贝/移动
    CarInfer(const CarInfer&) = delete;
    CarInfer& operator=(const CarInfer&) = delete;
    CarInfer(CarInfer&&) = delete;
    CarInfer& operator=(CarInfer&&) = delete;

    ~CarInfer() {
        stop();
        // plateDetPtr_/plateRecPtr_ 为 unique_ptr，无需手动 delete
    }

private:
    CarInfer() = default;

private:
    // ===== 原有公共字段中迁移到单例内的状态 =====
public:
    std::unique_ptr<std::thread> thread_main;
    std::unique_ptr<std::thread> thread_main2;
    std::string alg_name;

private:
    // 运行标志
    std::atomic<bool> APP_Status{true};
    std::atomic<bool> b_init{false};
    std::atomic<bool> started_{false};

    // 车识别（成员，避免 run() 内局部遮蔽）
    detYOLO infer;
	mutable std::mutex car_infer_mu_;  // 保护 plateDetPtr_ / plateRecPtr_ 推理

    // 车牌识别（改为智能指针）
    std::unique_ptr<lmYOLO> plateDetPtr_;
    std::unique_ptr<recCarPlate> plateRecPtr_;
	mutable std::mutex plate_infer_mu_;  // 保护 plateDetPtr_ / plateRecPtr_ 推理

    // ===== 单例 & 一次性初始化标记 =====
    static std::unique_ptr<CarInfer> instance_;
    static std::once_flag init_flag_;
    static std::once_flag model_flag_;
};

// ===== 静态成员定义 =====
std::unique_ptr<CarInfer> CarInfer::instance_;
std::once_flag CarInfer::init_flag_;
std::once_flag CarInfer::model_flag_;



#endif
