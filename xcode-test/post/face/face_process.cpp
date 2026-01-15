#include "face_process.h"


#include <stdio.h>
#include <vector>
#include "string.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "post/post_util.h"


#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

extern int G_port;
extern std::string sTypeName, sWebSockerUrl,sHttpUrl;
extern std::mutex g_lock_json;
extern std::shared_ptr<Json::Value> g_value;
extern std::shared_ptr<Json::Value> g_calc_switch;
const double EPS = 0.0000001;


extern float MaxConfLimit;

extern FaceServer *faceDb;

#ifdef STRANGERDB
extern StrangeServer* strangerDb;
#endif

using namespace std;

Postprocess_comm::Postprocess_comm(string xx)
{
    rtsp_path = xx;
    status = true;
    time_start_dict1 = 0;
    ntime = 0;
    facetool = new faceRec();
    facetool->init("");
    send_list = std::make_shared<BlockingQueue<sendMsg>>(3);
    t_sender = new std::thread(&Postprocess_comm::sendThread, this);
}

Postprocess_comm::~Postprocess_comm()
{
    status = false;
    
    
    if (t_sender != nullptr && t_sender->joinable())
    {
        t_sender->join();
        delete t_sender;
    }

    delete facetool;
    facetool = NULL;
    std::cout<<"~Postprocess()"<<std::endl;
}



std::string Postprocess_comm::makeSendMsg(std::shared_ptr<InferOutputMsg> data,Json::Value jresult,Json::Value jwarn,Json::Value roi,Json::Value jstranger,int alarm)
{
  
    std::string s;
    std::string img64,drawImage64;
    // gettimeofday(&tv1,NULL);
    if( data->rawImage.empty())
    {
        std::cout<<"imencode mat is empty! danger!"<<std::endl;
        return nullptr;
    }
    cv::Mat dstimg;
    int dstw,dsth;
    if(data->rawImage.cols != 1280)
    {           
        int w = 1280, h = 0;
        h = data->rawImage.rows * 1280 /data->rawImage.cols;
        if(h%2)
            h++;
        cv::resize(data->rawImage, dstimg, cv::Size(w, h), cv::INTER_AREA);
        cv_encode_base64(dstimg, img64);
        dstw = w;
        dsth = h;
    }
    else
    {
        cv_encode_base64(data->rawImage, img64);
        dstw = data->rawImage.cols;
        dsth = data->rawImage.rows;
    }

    if(alarm)
    {
        if(!jresult.isNull())
        {
            for(int i = 0; i< jresult.size(); i++)
            {
                cv::Point p1(jresult[i]["boxes"][0].asInt(), jresult[i]["boxes"][1].asInt());
                cv::Point p2(jresult[i]["boxes"][2].asInt() , jresult[i]["boxes"][3].asInt());
                cv::rectangle(dstimg, p1, p2, cv::Scalar(0, 255, 0), 2); //bgr
            }
        }
        if(!jwarn.isNull())
        {
            for(int i = 0; i< jwarn.size(); i++)
            {
                cv::Point p1(jwarn[i]["boxes"][0].asInt(), jwarn[i]["boxes"][1].asInt());
                cv::Point p2(jwarn[i]["boxes"][2].asInt() , jwarn[i]["boxes"][3].asInt());
                cv::rectangle(dstimg, p1, p2, cv::Scalar(0, 0, 255), 2);
            }
        }
        

        if(roi.size()>0 && roi[0].size()>0 && roi[0][0].isArray())//多roi
        {
            for(int j = 0; j <roi.size(); j++)
            {
                std::vector<cv::Point> points;
                for(int i = 0; i< roi[j].size(); i++)
                {
                    points.push_back(cv::Point((int)roi[j][i][0].asFloat(), (int)roi[j][i][1].asFloat())); 
                }
                if(roi[j].size())
                    cv::polylines(dstimg, points, true, cv::Scalar(255, 0, 0), 8);
            }
        }
        else//单roi
        {
            std::vector<cv::Point> points;
            for(int i = 0; i< roi.size(); i++)
            {
                points.push_back(cv::Point((int)roi[i][0].asFloat(), (int)roi[i][1].asFloat())); 
            }
            if(roi.size())
                cv::polylines(dstimg, points, true, cv::Scalar(255, 0, 0), 8);
        }
        
        cv_encode_base64(dstimg, drawImage64);//20
    }
        

    Json::Value j;
    j["type"] = sTypeName; 
    j["cam"]= rtsp_path;
    j["time"] = (int)time(0);
    j["alarm"] = alarm;
    j["size"].append(dstw);
    j["size"].append(dsth);
    if(jresult.isNull())
        j["result"].resize(0);
    else
        j["result"] = jresult;
    if(jwarn.isNull())
        j["warn"].resize(0);
    else
        j["warn"] = jwarn;
    if(jstranger.isNull())
        j["stranger"].resize(0);
    else
        j["stranger"] = jstranger;
    // j["video"] = recNameBuf;
    Json::Value jout;
    jout["data"] = j;
    s = jout.toStyledString();
    s.pop_back();
    s.pop_back();
    s.pop_back();
    s.pop_back();
    s.append(",\"img\":\"");
    s.append(img64);
    if(alarm)
    {
        s.append("\",\"drawImg\":\"");
        s.append(drawImage64);
    }
    s.append("\"},\"event\":\"pushVideo\"}");
    return s;
}

std::string Postprocess_comm::Process(std::shared_ptr<InferOutputMsg> data,int &isAlarm)
{

    g_lock_json.lock();
    Json::Value G_config = Get_Rtsp_Json(g_value,rtsp_path)["config"];
    Json::Value Roi = Get_Rtsp_Json(g_value,rtsp_path)["roi"];
    g_lock_json.unlock();
    
    std::time_t t = std::time(0);
    struct tm* st = gmtime(&t);

    int alarm = 0;


    Json::Value jwarn,jresult,jstranger;
    if(statusInTimes(G_config["work_time"]))
    {
            
        float score = G_config["score"].asFloat();
        int limArea = G_config["area"].asInt();
        // float conf = G_config["conf"].asFloat() * 1000.0;
        // float conf = G_config["conf"].asFloat()*2.0;
        float conf = (1 - G_config["conf"].asFloat())*2.0; //rk 相似度和距离相反



        
        if(oldtime < 0)//初始化
            oldtime = G_config["time"].asFloat();
        if(stranger_alarm_time < 0)
            stranger_alarm_time = G_config["stranger_alarm_time"].asFloat();

        if(!(abs(oldtime - G_config["time"].asFloat()) < EPS))
        { 
            std::cout<<"--[INFO] time is reset:"<<sTypeName<<rtsp_path<<std::endl;
            oldtime = G_config["time"].asFloat();
            time_interval = 0;
            time_start_dict1 = t;
            is_alarmed = 0;
            faceIdTimeMap.clear();
        }

        if(!(abs(stranger_alarm_time - G_config["stranger_alarm_time"].asFloat()) < EPS))
        { 
            std::cout<<"--[INFO] stranger_alarm_time is reset:"<<sTypeName<<rtsp_path<<std::endl;
            stranger_alarm_time = G_config["stranger_alarm_time"].asFloat();
            time_interval = 0;
            time_start_dict1 = t;
            is_alarmed = 0;
        }


        if(time_start_dict1 == 0)
        {
            time_start_dict1 = t;
            time_interval = t;
        }


        std::vector<cv::Point> contours;

        int previewH = 1280.f/(float)data->rawImage.cols * (float)data->rawImage.rows;
        if(previewH%2 != 0)
            previewH++;
        for (int i = 0; i < Roi.size(); i++)
        {
            //预览为1280分辨率，推理是原图，需要变换 ROI            
            int x = (float)Roi[i][0].asInt() * ((float) data->rawImage.cols / (float) 1280.f);
            int y = (float)Roi[i][1].asInt() * ((float) data->rawImage.rows / (float) previewH);
            contours.push_back(cv::Point(x, y));
        }

        vector<faceInfo> ans;

        std::vector<Yolov5Face_BoxStruct> face_result;
        std::vector<int> delNums;
        std::vector<float> input;
        int querySize;

        facetool->infer(data->rawImage,contours,face_result,delNums,input,querySize,score,limArea);
        
        faceDb->search(ans,face_result,delNums,input,querySize,2);

        if(ans.size() > 0)
        {                         
                    reDel:
                    for(auto it = faceIdTimeMap.begin(); it != faceIdTimeMap.end(); it++)
                    {
                        if(t - it->second > G_config["target_continued_time"].asInt()+3)
                        {
                            // printf("DEL it : %d\n",it->first);
                            faceIdTimeMap.erase(it->first);
                            goto reDel;
                        }
                    }
                    bool hadStranger = false;
                    #ifdef STRANGERDB
                    int inputIndex = -1;
                    int strangerId = -1;
                    #endif
                    for(int i = 0;i <ans.size(); i++)
                    {
                        #ifdef STRANGERDB
                        if(ans[i].fileName != "")
                            inputIndex++;
                        #endif

                        auto it = faceIdTimeMap.find(ans[i].id);
                        int type = -1;
                        int whatPeo = 0; // 0 = 数据库里的人，1 = 陌生人，2 = 不达标人
                        
                        if(ans[i].id != -2 && it == faceIdTimeMap.end() && conf >= ans[i].conf) // id 为-2代表数据库为空
                        {
                            type = 1;//还没开始计时
                            if(ans[i].id != -1)
                            {
                                whatPeo = 0;
                                faceIdTimeMap[ans[i].id] = t;
                            }
                            else
                            {
                                whatPeo = 2;
                                type = 3;//人脸质量不达标
                            }
                                
                        }
                        else
                        {    
                            if(ans[i].conf >  MaxConfLimit || ans[i].id == -2)
                            {
                                type = 2; 
                                whatPeo = 1;//陌生人
                                ans[i].fileName = "";
                                
                                ntime = t;
                                if( t - time_start_dict1 > G_config["target_continued_time"].asInt())
                                {
                                    hadStranger = true;
                                    type = 0; 
                                    
                                    // cout<<"time:"<<time_start_dict1<<",?:"<<t<<",stt:"<<stranger_alarm_time * 60<<"\n";
                                    // time_interval =  (time_t) (t + (oldtime * 60.0));
                                    time_interval =  (time_t) (t + (stranger_alarm_time * 60.0));
                                    alarm = 1;
                                    is_alarmed = 1;

                                    #ifdef STRANGERDB
                                    std::vector<float> strangerSearchInput;
                                    vector<float> SD;vector<idx_t> SI;
                                    for(int i = inputIndex*512; i < inputIndex*512+512; i++)
                                    {
                                        strangerSearchInput.push_back(input[i]);
                                    }
                                    strangerId = strangerDb->add(strangerSearchInput);
                                    printf("[warning] add a stranger face %d!\n", strangerId);
                                    #endif
                                }
                            }
                            else if(conf < ans[i].conf)
                            {
                                whatPeo = 2; //2 = 不达标人
                                type = 3;//人脸质量不达标
                                ans[i].fileName = "";
                            }
                            else if(t - it->second > G_config["target_continued_time"].asInt()) // 熟人
                            {

                                whatPeo = 0;
                                type = 0;//报警
                                alarm = 1;
                                it->second =  (time_t) (t + (oldtime * 60.0));
                                time_interval =  (time_t) (t + (oldtime * 60.0));
                                
                            }
                            else // 熟人
                            {
                                whatPeo = 0;
                                type = 1; //不报警
                            }
                            
                        }
                        Json::Value jboxone2;
                        Json::Value jboxone;                     
                        int jx = (float)ans[i].x * ((float) 1280.f / (float) data->rawImage.cols);
                        int jw = (float)ans[i].w * ((float) 1280.f / (float) data->rawImage.cols);
                        int jy = (float)ans[i].y * ((float) (float) previewH  / data->rawImage.rows);
                        int jh = (float)ans[i].h * ((float) (float) previewH  / (float) data->rawImage.rows);
                        // int jx = (float)ans[i].x * ((float)  data->rawImage.cols / (float) 1280.f);
                        // int jw = (float)ans[i].w * ((float) data->rawImage.cols / (float) 1280.f);
                        // int jy = (float)ans[i].y * ((float) data->rawImage.rows  /(float) previewH);
                        // int jh = (float)ans[i].h * ((float) data->rawImage.rows / (float) previewH);
                        jboxone.append(jx);
                        jboxone.append(jy);
                        jboxone.append(jx+jw);
                        jboxone.append(jy+jh);
                        
                        jboxone2["boxes"]= jboxone;
                        jboxone2["whatPeo"]= whatPeo;
                        #ifdef STRANGERDB
                        if(whatPeo == 1 && type == 0)
                            jboxone2["strangerId"]= strangerId;
                        else
                            jboxone2["spId"] = (int)ans[i].spId;
                        #endif
                        jboxone2["score"] = ans[i].score;
                        // jboxone2["conf"] = ans[i].conf;//距离
                        jboxone2["conf"] = int((1-ans[i].conf/2) * 100) / 100.0   ;//相似度
                        jboxone2["fileName"] = ans[i].fileName;
                        

                        // printf("detect face type %d [%s] [%d] [%03f]\n",type,ans[i].fileName.c_str(),ans[i].id,ans[i].conf);

                        if(type == 0)//报警
                        {
                            if(whatPeo == 1)
                                jboxone2["dur_time"] = (int) (t - time_start_dict1);//持续时间;
                            else
                                jboxone2["dur_time"] = (int) (t - faceIdTimeMap[ans[i].id]);//持续时间
                            jwarn.append(jboxone2);
                        }
                        else if (type == 2)
                        {
                            jboxone2["dur_time"] = (int) (t - time_start_dict1);//持续时间;
                            jresult.append(jboxone2);
                        }
                        else if(type == 1)//不报警，持续时间不满足
                        {
                            jboxone2["dur_time"] = (int) (t - faceIdTimeMap[ans[i].id]);//持续时间
                            jresult.append(jboxone2);
                        }
                        else //人脸质量不达标
                        {
                            jboxone2["dur_time"] = 0;
                            jresult.append(jboxone2);
                        }
                    }
                    if(hadStranger) //所有陌生人用同一个时间
                    {
                        time_start_dict1 = (time_t) (t + (oldtime * 60.0));
                    }
                
               
        }

        
    }

    g_lock_json.lock();
    Json::Value temp = *g_calc_switch;
    g_lock_json.unlock();
    string str1 = (temp)["data"]["rtsp"].asString();
    string str2 = rtsp_path;
    string str3 = (temp)["data"]["alg_id"].asString();

    bool switchIsON = ((str1 == str2) && str3 == sTypeName);
    std::string s;
    if(switchIsON || (!switchIsON && alarm == 1) )
    {
        s = makeSendMsg(data,jresult,jwarn,Roi,jstranger,alarm);

        isAlarm = alarm;

        if(alarm)
            std::cout<<"[INFO]: "<<sTypeName<<"--ws send from rtsp : "<<rtsp_path<<"---"<<alarm<<" alarm:"<<alarm<<endl;

    }
    return s;
}

void Postprocess_comm::sendThread()
{
    
//----------------------------------------------------------------------------
    while(this->status)
    {
        try{
            timeval t1,t2;
            // ws
            std::string wst = sWebSockerUrl; // ws://127.0.0.1:9698/ws
            std::cout<<"ws url:"<<sWebSockerUrl<<std::endl;
            int index_1 = wst.find("//");
            wst = wst.substr(index_1+2,wst.size());
            index_1 = wst.find("/ws");
            wst = wst.substr(0,index_1);

            index_1 = wst.find(":");
            std::string IP = wst.substr(0,index_1);

            // auto const port = "9698";
            int divNum = (G_port % 10);
            int wsPort = G_port - divNum +8;
            auto const port = to_string(wsPort);
            
            net::io_context ioc;
            tcp::resolver resolver{ioc};
            websocket::stream<tcp::socket> ws{ioc};
            auto const results = resolver.resolve(IP, port);
            auto ep = net::connect(ws.next_layer(), results);
            IP += ':' + std::to_string(ep.port());
            ws.set_option(websocket::stream_base::decorator(
                [](websocket::request_type &req)
                {
                    req.set(http::field::user_agent,
                            std::string(BOOST_BEAST_VERSION_STRING) +
                                " websocket-client-coro");
                }));
            ws.handshake(IP, "/ws");

            int sendCnt = 0;
            while(this->status)
            {
                if(send_list->GetSize()>0)
                {
                    sendMsg temp;
                    if(send_list->Pop(temp) == 0)
                    {
                        ws.write(net::buffer(std::string(temp.msg)));                        
                    }
                    else
                        usleep(10000);
                        
                }
                else
                    usleep(10000);
            }
                ws.close(websocket::close_code::normal);
            }

        catch (const std::exception &e)
        {
            std::cerr << "out error: " << e.what() << '\n';
        }
    }
    send_list->Clear();
    // send_list.reset();
    std::cout << "send thread down" << std::endl;
}