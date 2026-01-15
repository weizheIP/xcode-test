#ifndef __GET_VIDEO_H__
#define __GET_VIDEO_H__
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <thread>
#include <unistd.h>
#include <iostream>
#include <numeric>
#include <atomic>
#include <mutex>
#include "post/queue.h"



#define __THREAD__

extern "C"
{
    #include "libavcodec/avcodec.h"
    #include "libavformat/avformat.h"
    #include "libswscale/swscale.h"
    // #include "libavdevice/avdevice.h"
    #include "libavutil/pixfmt.h"
    #include "libavutil/imgutils.h"
    #include "libavutil/opt.h"
    #include <libavdevice/avdevice.h>
    #include <libavutil/pixdesc.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/opt.h>
    #include <libavutil/avassert.h>
}

#include <boost/thread/pthread/shared_mutex.hpp>

typedef boost::shared_lock<boost::shared_mutex> read_lock;
typedef boost::unique_lock<boost::shared_mutex> write_lock;

#include "common/MyLock/write_priotity_lock.h"

static void sw_resize(cv::Mat &tempImg1)
{
    // if(tempImg1.cols != 1280 && !tempImg1.empty())
    // {   cv::Mat dstimg;
    //     int w = 1280, h = 0;
    //     h = tempImg1.rows * 1280 /tempImg1.cols;
    //     //std::cout<<tempImg1.cols<<tempImg1.rows<<"-"<<dstw<<":"<<dsth<<std::endl;
    //     if(h%2)
    //         h++;
    //     cv::resize(tempImg1, dstimg, cv::Size(w, h), cv::INTER_AREA);
    //     tempImg1 = dstimg.clone();
    //     dstimg.release();
    // }
    if(tempImg1.empty())
        return;
    if(tempImg1.cols>tempImg1.rows){
        int hh = tempImg1.rows * 1280 / tempImg1.cols;
        if(hh%2)
            hh++;
        cv::resize(tempImg1, tempImg1, cv::Size(1280, hh));
    }else{
        int ww = tempImg1.cols * 1280 / tempImg1.rows;
        if(ww%2)
            ww++;
        cv::resize(tempImg1, tempImg1, cv::Size(ww, 1280));
    }
}

class VideoDeal
{
public:

    bool isPause = false;
    void pause(){isPause = true;};
    void start(){isPause = false;};
    int statue(){return isPause;};
    std::string errMsg="";

    int testH,testW;
    VideoDeal(){;}
    

    int getframe(cv::Mat &mat);
    AVPacket* getPacketFromQue();
    //录像
    int encode_init(const char *outfile);
    void encode_deinit();
    void av_packet_rescale_ts(AVPacket *pkt, AVRational src_tb, AVRational dst_tb);
    int WritePacket(AVPacket* packet);
    void startRec(char *filename);
    
    ~VideoDeal();
    //输入
    AVFormatContext *pFormatCtx;
    AVCodecContext* pCodecCtx;
    bool hw_decode = true;
    // #ifdef _SelCard
    int _deviceId;
    VideoDeal(std::string videoPath,int deviceId=0);
    int hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type,int deviceId=0);
    // #else
    // VideoDeal(std::string videoPath);
    // int hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type);
    // #endif
    AVBufferRef *hw_device_ctx = NULL;
    enum AVPixelFormat hw_pix_fmt;

    int cams_count = 0; // 全局摄像头数量
    
    //rtsp流参数
    std::string g_file_path;
    std::string video_path;
    int width;
    int height;
    int fps;
    //录像参数
    int alarFrameCnt;
    int waitFrameCnt = 999;
    bool isRecing = false;
    int64_t recfirFPts =0;
    bool isFile = false;

    //2、输出
    AVFormatContext *oc = NULL;
    int videoStream_best;

    bool bvideoSave = false;
    bool bvideoRead = false;

    volatile bool status;
    bool isReSize;

    void getter();
    void alarmRecer(char *filename);
    
    std::shared_ptr<BlockingQueue<AVPacket*>> Rec_Frame_lists;

    std::shared_ptr<BlockingQueue<AVFrame*>> frame_list;

    std::shared_ptr<BlockingQueue<AVPacket*>> packet_list;

    std::shared_ptr<BlockingQueue<AVFrame*>> hwFrame_list;

    std::shared_ptr<BlockingQueue<cv::Mat>> resize_in_list;
    std::shared_ptr<BlockingQueue<cv::Mat>> resize_out_list;
    void getPacket();
    void getHwFrame();
    void resize_frame();
    int end_of_stream = 1;

    cv::Mat g_img_decode;
    std::mutex mlock;
    std::mutex tlock;
    volatile std::atomic<bool> b_getter;
    std::thread *t_getter = nullptr;
    std::thread *t_alarRecer = nullptr;

    int main_ffmpeg(const char * file_path);
    int ReStart();


    int getNewestFrame(cv::Mat &mat, int &index);
    boost::shared_mutex m_mat_lock;
    write_priotity_lock mMatLock;
    cv::Mat m_cv_mat;
    int mIndex = -1;
    std::vector<int> useTimeArr;
    
};

#endif

