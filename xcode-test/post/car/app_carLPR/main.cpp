

#include <map>
#include "unistd.h"
#include <memory.h>
#include <mutex>
#include <iostream>
#include <signal.h>
#include "license.h"
#include "httpprocess.h"
#include <mutex>
#include <map>
#include "unistd.h"
#include <memory.h>
#include <atomic>
#include <sys/wait.h>
#include "Track/inc/SORT.h"

#include "prefer/get_videoffmpeg.hpp"
#include "post/queue.h"
#include "post/post_util.h"
#include "post_process.h"
#include "post/model_base.h"


using namespace std;

Json::Value g_value;
Json::Value g_calc_switch;
std::mutex g_lock_json;

string sTypeName = "cv_large_car_detection";
string sHttpUrl = "http://127.0.0.1:9696";
string sHttpPath = "/plg/tdds/calc/rtsp";
string sWebSockerUrl = "ws://127.0.0.1:9698/ws";
string pushVideoPath = "/api/ws/getPushVideo";

BlockingQueue<message_img> queue_pre;
BlockingQueue<message_lpr_img> queue_lpr_pre;
std::map<std::string, std::shared_ptr<BlockingQueue<message_det>>> queue_post;
std::map<std::string, std::shared_ptr<BlockingQueue<message_lpr>>> queue_lpr_post;
std::mutex queue_post_lock;
std::mutex queue_lpr_post_lock;


std::vector<std::string> lic_algs;
std::string lic_algs_all;



class AlgsInfer
{
public:
    bool APP_Status = true;
    thread *thread_main;
    thread *thread_main2;

    AlgsInfer(std::string rtsp_path_)
    {
        thread_main = new thread(&AlgsInfer::run, this);
        thread_main2 = new thread(&AlgsInfer::run2, this);
    }

    void run()
    {
        message_img premsg;

        detYOLO infer;
        // infer.init("m_enc.pt",1,1); //只用npu核1，多次载入导致rknn_set_core_mask报错
        infer.init("m_enc.pt", 1, 3); // 核3
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
        plateDetPtr_->init("det.rknn", 0);
        plateRecPtr_->init("rec.rknn");
        while (APP_Status)
        {
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
            msg.lprCode = code;
            queue_lpr_post_lock.lock();
            std::shared_ptr<BlockingQueue<message_lpr>> aa = queue_lpr_post[premsg.videoIndex];
            queue_lpr_post_lock.unlock();
            if (aa->Push(msg, true) != 0)
            {
                usleep(1000);
            }
        }
    }
    void stop()
    {
        APP_Status = false;
        if (thread_main->joinable())
            thread_main->join();
        delete thread_main;
        if (thread_main2->joinable())
            thread_main2->join();
        delete thread_main2;
    }
    ~AlgsInfer()
    {
        stop();
    }
};

class APP
{
public:
    
    VideoDeal **video;   
    bool APP_Status = true;
    thread *thread_main;
    std::string rtsp_path;
    bool isRunning = false;

    void app_process()
    {
        Postprocess_comm m_post(rtsp_path,sTypeName);
        if (m_post.status == false)
            return;
        timeval tv0, tv1, tv2, tv3;

        VideoDeal *videoPtr = new VideoDeal(rtsp_path);
        video = &videoPtr;

        if (!(*video)->status)
        {
            std::cout << "[Error] Open Error," << rtsp_path << std::endl;
        }

        cv::Mat srcimg;

        int getRet = 0;
        int resetC = 0;
        int sendCnt = 0;

    
        vector<int> ms1, ms2, ms3, ms4;
        vector<float> fps5;


        while (APP_Status)
        {
            // 1. 解码
            gettimeofday(&tv0, NULL);
            getRet = (*video)->getframe(srcimg);
            if (getRet)
            {
                if (getRet == 1)
                {
                    if ((*video)->fps <= 0)
                        usleep(1000 * 20);
                    else
                        usleep(1000 * (1000 / (*video)->fps));
                    continue;
                }

                if (!APP_Status)
                    break;
                std::cout << "[INFO] Video play done && Video reload sleep 5 sec:" << rtsp_path << std::endl;
                sleep(5);                
                delete (*video);
                (*video) = nullptr;
                (*video) = new VideoDeal(rtsp_path);

                sleep(2);
                if (!(*video)->status)
                {
                    std::cout << "[Error] Open Error," << rtsp_path << std::endl;
                    continue;
                }
                continue;
            }
            if (srcimg.empty())
                continue;
            gettimeofday(&tv1, NULL);

            // 2. 推理
            message_img premsg;
            premsg.frame = srcimg;
            premsg.videoIndex = rtsp_path;
            if (queue_pre.Push(premsg) != 0)
            {
                usleep(1000);
            }

            vector<BoxInfo> carBoxInfoVec;          
            message_det postmsg;
            queue_post_lock.lock();
            std::shared_ptr<BlockingQueue<message_det>> aa = queue_post[rtsp_path];
            queue_post_lock.unlock();      
            if (aa->IsEmpty()) //不阻塞
        {
            usleep(1000);
            continue;
        }        
            if (aa->Pop(postmsg) != 0)
            {
                usleep(1000);
                continue;
            }
            carBoxInfoVec = postmsg.detectData;

            // 3. 后处理
            std::shared_ptr<InferOutputMsg> data = std::make_shared<InferOutputMsg>();
            int ret = 0;
            gettimeofday(&tv2, NULL);
            data->rawImage = postmsg.frame;
            data->result = carBoxInfoVec;
            sendMsg msg;
            processInfo info;
            
            std::string send_result  = m_post.process(data, info);
            gettimeofday(&tv3, NULL);
            msg.msg = send_result;
            
            if (msg.msg.empty())
            {
                struct tm st = {0};
                localtime_r(&tv3.tv_sec, &st);
                if (sendCnt++ >= 30)
                {
                    sendCnt = 0;
                    std::cout << "prinf from app, "
                              << "FPS:" << t_get_vec_aver(ms1)
                              << ",getframe:" << t_get_vec_aver(ms2)
                              << "ms,detect:" << t_get_vec_aver(ms3)
                              << "ms,process:" << t_get_vec_aver(fps5)
                              << "ms,nowTime:" << st.tm_year + 1900 << "-" << st.tm_mon + 1 << "-" << st.tm_mday << " " << st.tm_hour << ":" << st.tm_min << ":" << st.tm_sec
                              << "rtsp:" << rtsp_path
                              << std::endl;
                }
                if (ms1.size() > 30)
                {
                    ms1.erase(ms1.begin());
                    ms2.erase(ms2.begin());
                    ms3.erase(ms3.begin());
                    fps5.erase(fps5.begin());
                }
                ms1.push_back((tv1.tv_sec - tv0.tv_sec) * 1000 + (tv1.tv_usec - tv0.tv_usec) / 1000);
                ms2.push_back((tv2.tv_sec - tv1.tv_sec) * 1000 + (tv2.tv_usec - tv1.tv_usec) / 1000);
                ms3.push_back((tv3.tv_sec - tv2.tv_sec) * 1000 + (tv3.tv_usec - tv2.tv_usec) / 1000);
                fps5.push_back(1.0 / (tv3.tv_sec - tv0.tv_sec + tv3.tv_usec / 1000000.0 - tv0.tv_usec / 1000000.0));
                srcimg.release();
                data.reset();
                continue;
            }

     
            if (m_post.send_list.get() != nullptr)
            {
                while (msg.msg.size() != 0 && m_post.send_list->Push(msg) != 0)
                {
                    usleep(30000);
                }
            }
            else
            {
                srcimg.release();
                data.reset();
                break;
            }
            srcimg.release();
            data.reset();
        }
   


        srcimg.release();
        if ((*video))
        {
            delete (*video);
        }
    }

    APP(std::string rtsp_path_) : rtsp_path(rtsp_path_)
    {
        APP_Status = true;
        run();
    }
    void run()
    {
        thread_main = new thread(&APP::app_process, this);
    }
    void stop()
    {
        APP_Status = false;
        if (thread_main->joinable())
            thread_main->join();
        delete thread_main;
    }
    ~APP()
    {
        stop();
    }
};




class Main
{
public:
    std::map<std::string, std::unique_ptr<APP>> APP_list;
    bool checkrtsplist(vector<string> l1, vector<string> l2)
    {
        if (l1.size() != l2.size())
            return false;
        vector<string>::iterator t1, t2;
        for (t1 = l1.begin(); t1 != l1.end(); t1++)
        {
            bool b_checkl = false;
            for (t2 = l2.begin(); t2 != l2.end(); t2++)
            {
                if (*(t2) == *(t1))
                {
                    b_checkl = true;
                    break;
                }
            }
            if (!b_checkl)
                return false;
        }
        return true;
    }
    void get_http_rtsp()
    {
        HttpProcess http(sHttpUrl);
        string data;
        vector<string> vrtsplist;

        HttpProcess http2(sHttpUrl);
        string pushVideoData;

        AlgsInfer a("");
        while (1)
        {
            try
            {              
                http.Get(sHttpPath, data);
                if (data.find("{") == std::string::npos)
                {
                    std::cout << "http get algs camsList error" << std::endl;
                    std::cout << data << std::endl;
                    sleep(2);
                    continue;
                }
                Json::Value value = http.Jsonparse(data)["data"];
                if (value.isNull())
                {
                    std::cout << "data2 error" << std::endl;
                    std::cout << value << std::endl;
                    sleep(2);
                    continue;
                }

                //---------------------------------------------------------
                http2.Get(pushVideoPath, pushVideoData);
                Json::Value value2 = http2.Jsonparse(pushVideoData)["data"];
                if (value2.isNull())
                {
                    std::cout << "data2 error" << std::endl;
                    std::cout << value2 << std::endl;
                    sleep(2);
                    continue;
                }
                if (!value.isArray() || !value2.isObject())
                {
                    std::cout << sTypeName << "--cams list or pushvideo is not a arr" << std::endl;
                    std::cout << value.asCString() << std::endl;
                    std::cout << value2.asCString() << std::endl;
                    sleep(2);
                    continue;
                }

                g_lock_json.lock();
                g_calc_switch = Json::Value(http.Jsonparse(pushVideoData));
                g_value = Json::Value(http.Jsonparse(data));
                g_lock_json.unlock();
                //---------------------------------------------------------
                vector<string> vrtsplist_u;
                for (int index = 0; index < value.size(); index++)
                {
                    vrtsplist_u.push_back(value[index]["rtsp"].asString());
                    if (std::count(vrtsplist.begin(), vrtsplist.end(), value[index]["rtsp"].asString()) == 0)
                    {
                        std::unique_ptr<APP> app;
                        app.reset(new APP(vrtsplist_u[index]));
                        APP_list[vrtsplist_u[index]] = (std::move(app));
                        queue_post_lock.lock();
                        queue_post[vrtsplist_u[index]] = std::make_shared<BlockingQueue<message_det>>(10);
                        queue_post_lock.unlock();
                        queue_lpr_post_lock.lock();
                        queue_lpr_post[vrtsplist_u[index]] = std::make_shared<BlockingQueue<message_lpr>>(10);
                        queue_lpr_post_lock.unlock();
                    }
                }
                if (!checkrtsplist(vrtsplist, vrtsplist_u) && vrtsplist.size() > 0) // 关闭没有的
                {
                    for (int index_i = 0; index_i < vrtsplist.size(); index_i++)
                    {
                        if (std::count(vrtsplist_u.begin(), vrtsplist_u.end(), vrtsplist[index_i]) == 0)
                        {
                            if (APP_list.count(vrtsplist[index_i]) != 0)
                            {
                                APP_list[vrtsplist[index_i]].reset();
                            }
                            queue_post_lock.lock();
                            queue_post.erase(vrtsplist[index_i]);
                            queue_post_lock.unlock();
                            queue_lpr_post_lock.lock();
                            queue_lpr_post.erase(vrtsplist[index_i]);
                            queue_lpr_post_lock.unlock();
                        }
                    }
                }
                vrtsplist = vrtsplist_u;
                sleep(1);
            }
            catch (...)
            {
                std::cout << "[Error] Http communicate Error" << std::endl;
                sleep(5);
            }
            usleep(500000);
        }
    }
};

int main(int argc, char **argv)
{
    bool isDaemon = false;
    std::string serverIp;

    if (argc == 4)
    {
        isDaemon = atoi(argv[3]);
        serverIp = argv[2];
        sTypeName = argv[1];
    }
    else
    {
        serverIp = "127.0.0.1";
        sTypeName = "license_plate_detection";
    }

    if (!checklic("../../civiapp/licens/license.lic"))
    {
        std::cout << "[Error] Check license failed!" << std::endl;
        return 0;
    }

    sHttpUrl = "http://" + serverIp + ":9696";
    sWebSockerUrl = "ws://" + serverIp + ":9698/ws";
    sHttpPath = "/plg/" + sTypeName + "/calc/rtsp";

    pid_t pid_main = getpid();
    pid_t pid;
    int status;

    if (isDaemon)
    {
        pid = fork();
    }

    while (isDaemon)
    {
        if (getpid() != pid_main)
            break;
        if (getpid() == pid_main)
        {
            pid_t result = waitpid(pid, &status, WNOHANG);
            if (result != 0) // pid不存在，创建新进程
            {
                std::cout << "[ERROR] pid:" << pid << "is dead ! restart !" << std::endl;
                pid = fork();
                result = waitpid(pid, &status, WNOHANG);
            }
        }
        else
            break;
        sleep(2);
    }

    Main demo;
    demo.get_http_rtsp();

 
    std::cout << "app stopped!" << std::endl;
}
