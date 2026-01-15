#ifndef FACE_DEMO
#define FACE_DEMO


#include "face_db.h"
#include "face_process.h"

// #include "config.h"
#include "license.h"
#include "httpprocess.h"

#include "prefer/get_videoffmpeg.hpp"
// #include "test_main/get_video.hpp"
#include <mutex>

#include <map>
#include "unistd.h"
#include <memory.h>
#include <atomic>

#include "Track/inc/SORT.h"
#include "post/queue.h"

#include <sys/wait.h>




class APP {
public:
    // bool APP_Status = true;
    std::atomic_bool APP_Status{true};
    thread *thread_main;
    std::string rtsp_path;
    // bool isRunning = false;

    void app_process();

    APP(std::string rtsp_path_):rtsp_path(rtsp_path_){
        APP_Status = true;
        run();
    }
    void run()
    {
        thread_main = new thread(&APP::app_process,this);
    }
    void stop(){
        APP_Status = false;
        if(thread_main->joinable())
            thread_main->join();
        delete thread_main;
    }
    ~APP(){
        stop();
    }
};



extern bool app_stopped;

class Main{
public:
    bool checkrtsplist(vector<string> l1,vector<string> l2);
    void get_http_rtsp();
};


#endif