#include <map>
#include "unistd.h"
#include <memory.h>
#include <mutex>

#include <sys/wait.h>
#include "faceDemo.h"

std::shared_ptr<Json::Value> g_value = std::make_shared<Json::Value>();
std::shared_ptr<Json::Value> g_calc_switch = std::make_shared<Json::Value>();
std::mutex g_lock_json;

string sTypeName = "face_detection";

string sHttpUrl = "http://192.168.18.59:9696";
string sHttpPath = "/api/camera/preview/rtsp";
string sWebSockerUrl = "ws://192.168.18.59:9696/ws";
std::string sLicense_path = "./license.lic";
string getPushVideoUrl = "http://127.0.0.1:9696";
string pushVideoPath = "/api/ws/getPushVideo";

// std::vector<std::string> lic_algs;
// std::string lic_algs_all;
// extern std::string device_id;

int G_port = 9696;

float MaxConfLimit = 1.72;


std::map<std::string,std::unique_ptr<APP>> APP_list;



FaceServer *faceDb= nullptr;

#ifdef STRANGERDB
StrangeServer *strangerDb= nullptr;
#endif

void APP::app_process()
{
Begin:
    Postprocess_comm m_post(rtsp_path);
    if(m_post.status == false)
        return;
    timeval tv0,tv1,tv2,tv3;
    VideoDeal* video = new VideoDeal(rtsp_path);
    std::cout<<"[INFO] open cam,"<<rtsp_path<<std::endl;
    

    if(!video->status)
    {
        std::cout<<"[Error] Open Error,"<<rtsp_path<<std::endl;
        //return;
    }

    
    int dt = 10;
    for(int i =0 ;i<dt;i++) //等线程启动，不做这个相机有概率开不起来
    {
        cv::Mat srcimg;
        if(video->getframe(srcimg)){
            // std::cout<<"Count Read Img"<<std::endl;
            usleep(100000);
        }
        else{
            break;
        }
    }
    int getRet = 0;
    int resetC = 0;
    int sendCnt =0;
    
    while(APP_Status)
    {
        cv::Mat srcimg;
        gettimeofday(&tv0, NULL);
        getRet = video->getframe(srcimg);
        // if(!getRet){         
        //     std::cout<<"[ERROR] Video play done && Video reload sleep 3 sec:"<<rtsp_path<<std::endl;       
        //     if(!APP_Status)
        //         break;
        //     std::cout<<"[ERROR] Video play done && Video reload sleep 3 sec:"<<rtsp_path<<std::endl;
        //     delete video;
        //     video = new VideoDeal(rtsp_path);
        //     sleep(3);      
        //     continue;
        // }
        if(getRet){
            if(getRet == 1)
            {
                //std::cout<<"[INFO] yuvlist is empty,"<<rtsp_path<<std::endl;
                if(video->fps <= 0)
                    usleep(1000*20);
                else
                    usleep(1000* (1000/video->fps));
                continue;
            }

            // if(!APP_Status)
            //     break;
            std::cout<<"[ERROR] Video play done && Video reload sleep 3 sec:"<<rtsp_path<<std::endl;
            delete video;
            sleep(2);
            video = new VideoDeal(rtsp_path);
            sleep(3);
            if(!video->status)
            {
                std::cout<<"[ERROR] Open Error,"<<rtsp_path<<std::endl;
                continue;
            }
            continue;
        }

        if(srcimg.empty())
            continue;
        std::shared_ptr<InferOutputMsg> data = std::make_shared<InferOutputMsg>();
        
        gettimeofday(&tv1, NULL);

        int ret = 0;
        
        gettimeofday(&tv2, NULL);
        data->frameWidth = srcimg.cols;
        data->frameHeight = srcimg.rows;
        
        
        data->rawImage = srcimg;

        std::string send_result;
        int isAlarm = 0;
        send_result = m_post.Process(data,isAlarm);
        gettimeofday(&tv3, NULL);
        if(sendCnt++ >= 30)
        {
            sendCnt = 0;
            std::cout <<"[FPS]:"<<1.0/(tv3.tv_sec-tv0.tv_sec+ tv3.tv_usec/1000000.0-tv0.tv_usec/1000000.0)
            <<",getframe:"<<(tv1.tv_sec - tv0.tv_sec)*1000 + (tv1.tv_usec - tv0.tv_usec)/1000
            <<" ms,detect:"<<(tv3.tv_sec - tv2.tv_sec)*1000 + (tv3.tv_usec - tv2.tv_usec)/1000
            <<" ms,rtsp:"<<rtsp_path
            <<std::endl;
        }

        if(send_result.empty())
        {            
            // srcimg.release();
            data.reset();
            continue;
        }
        sendMsg msg;
        msg.msg = send_result;
        msg.tvN[0] = tv0;
        msg.tvN[1] = tv1;
        msg.tvN[2] = tv2;
        msg.tvN[3] = tv3;
        if(m_post.send_list.get() != nullptr)
        {
            if(isAlarm)//防止警报信息被扔掉
            {
                // std::cout<<"[ERROR] xxxxxxxxxxxxxxxxxxxxxxxxx ALARM **********************************\n";
                while(m_post.send_list->Push(msg) != 0)
                    usleep(20*1000);
            }
            else
                m_post.send_list->Push(msg);
        }
        // else
        // {
        //     // srcimg.release();
        //     data.reset();
        //     break;
        // }
        // srcimg.release();
        data.reset();
    }
    // srcimg.release();
    delete video;
    std::cout<<"[INFO] Release cam: "<<rtsp_path<<std::endl;
}

bool Main::checkrtsplist(vector<string> l1,vector<string> l2)
{
    if(l1.size()!=l2.size())return false;
    vector<string>::iterator t1,t2;
    for(t1 = l1.begin();t1!=l1.end();t1++)
    {
        bool b_checkl=false;
        for(t2=l2.begin();t2!=l2.end();t2++)
        {
            if(*(t2)==*(t1))
            {
                b_checkl = true;
                break;
            }
        }
        if(!b_checkl)return false; 
    }
    return true;
}

void Main::get_http_rtsp()
{

    faceDb = new FaceServer("../../civiapp/faceDatabase");

    #ifdef STRANGERDB
    strangerDb = new StrangeServer();
    #endif

    HttpProcess http(sHttpUrl);
    string data;
    std::time_t tv2=0,tv3=0;
    vector<string> vrtsplist;

    HttpProcess http2(getPushVideoUrl);
    string pushVideoData;

    while(1)
    {
        try{
            // if (app_stopped){
            //     break;
            // }
            // std::time_t tv1,tv2,tv3;
            tv3 = std::time(0);
            if(tv3 - tv2 >= 60)
            {
                tv2 = tv3;
                // std::time_t tmp = tv2 - tv1;
                // int day,hour,min,sec;
                // day = tmp/86400;
                // hour = (tmp%86400)/3600;
                // min  = (tmp%3600)/60;
                // sec = (tmp%60);
                // std::cout<<"\033[35m"<<"\e[1m"<<"Program running time >> "<<"day:"<<day<<",hour:"<<hour<<",min:"<<min<<",sec:"<<sec<<"\e[0m"<<"\n";
                std::cout<<"[INFO] time main ------------------------------ "<<"\n";
            }

            http.Get(sHttpPath,data);
            if(data.find("{") == std::string::npos) 
            {
                std::cout<<"http get algs camsList error"<<std::endl;
                std::cout<<data<<std::endl;
                sleep(2);
                continue;
            }
            Json::Value value = http.Jsonparse(data)["data"];
            if(value.isNull())
            {
                std::cout<<"data2 error"<<std::endl;
                std::cout<<value<<std::endl;
                sleep(2);
                continue;
            }
            // std::cout<<value.toStyledString()<<std::endl;

//---------------------------------------------------------
            http2.Get(pushVideoPath,pushVideoData);
            Json::Value value2 = http2.Jsonparse(pushVideoData)["data"];
            if(value2.isNull())
            {
                std::cout<<"data2 error"<<std::endl;
                std::cout<<value2<<std::endl;
                sleep(2);
                continue;
            }
            if(!value.isArray() || !value2.isObject())
            {
                std::cout<<sTypeName<<"--cams list or pushvideo is not a arr"<<std::endl;
                std::cout<<value.asCString()<<std::endl;
                std::cout<<value2.asCString()<<std::endl;
                sleep(2);
                continue;
            }

            g_lock_json.lock();
            g_calc_switch.reset(new Json::Value(http2.Jsonparse(pushVideoData)));
            g_value.reset(new Json::Value(http.Jsonparse(data)));
            g_lock_json.unlock();
//---------------------------------------------------------
            vector<string> vrtsplist_u;
            for(int index = 0;index<value.size();index++)
            {
                vrtsplist_u.push_back(value[index]["rtsp"].asString());
                // char push_name[100]={0};
                // snprintf(push_name,100,"%s%s",rawpath.c_str(),value[index]["rtsp"].asString().c_str());
                if(std::count(vrtsplist.begin(),vrtsplist.end(),value[index]["rtsp"].asString())==0)
                {
                    std::unique_ptr<APP> app;
                    app.reset(new APP(vrtsplist_u[index]));
                    APP_list[vrtsplist_u[index]] = (std::move(app));
                    sleep(2); //onnx和ts不能太快
                }
            }
            if(!checkrtsplist(vrtsplist,vrtsplist_u) && vrtsplist.size()>0) // 关闭没有的
            {
                for(int index_i = 0;index_i<vrtsplist.size();index_i++){
                    if(std::count(vrtsplist_u.begin(),vrtsplist_u.end(),vrtsplist[index_i]) == 0)
                    {
                        if(APP_list.count(vrtsplist[index_i])!=0)
                        {
                            APP_list[vrtsplist[index_i]].reset();
                        }
                    }
                }
            }
            vrtsplist = vrtsplist_u;
            usleep(200000);
        }
        catch(...)
        {
            std::cout<<"[Error] Http communicate Error"<<std::endl;
            sleep(5);

        }
        usleep(500000);
    }
}

#include <iostream>
#include <signal.h>
bool app_stopped = false;


using namespace std;
int main(int argc,char** argv)
{    

    std::string serverIp ="127.0.0.1";
    bool isDaemon = false;
    if(argc == 4)
    {
        MaxConfLimit = atof(argv[3]);
        isDaemon = atoi(argv[2]);
        serverIp = argv[1];       
    }
    sHttpUrl = "http://"+serverIp+":"+ to_string(9696);
    sWebSockerUrl = "ws://"+serverIp+":"+ to_string(9698)+"/ws";        
    getPushVideoUrl = "http://"+serverIp+":"+ to_string(9696);
    sHttpPath = "/plg/"+sTypeName+"/calc/rtsp";     

#ifndef DEED_DEBUG
    if(!checklic("../../civiapp/licens/license.lic")){
        std::cout<<"[Error] Check license failed!"<<std::endl;
        return 0;
    }
#endif


    pid_t pid_main = getpid();
    pid_t pid;
    int status;

    if(isDaemon)
    {
       pid = fork();
    }

    while (isDaemon)
    {
        if(getpid() != pid_main) break;
        if (getpid() == pid_main)
        {
            pid_t result = waitpid(pid, &status, WNOHANG);
            if(result != 0)//pid不存在，创建新进程
            {
                std::cout << "[INFO] pid" << pid << "Done!" << std::endl;
                pid = fork();
                result = waitpid(pid, &status, WNOHANG);
            }
        }
        else break;
        sleep(1);
    }

    Main demo;
    demo.get_http_rtsp();

    // while(true){
	// 	// std::cout << "while loop..." << std::endl;
	// 	if (app_stopped){
	// 		break;
	// 	}
	// }
	std::cout << "app stopped!" << std::endl;
}

