#ifndef __MP4_MUXER_H__
#define __MP4_MUXER_H__
#include <unistd.h>
#include <iostream>
#include <thread>
#include <mutex>
extern "C"
{
    #include "libavcodec/avcodec.h"
    #include "libavformat/avformat.h"
    #include "libswscale/swscale.h"
    // #include "libavdevice/avdevice.h"
    #include "libavutil/pixfmt.h"
    #include "libavutil/imgutils.h"
    #include "libavutil/opt.h"
    // #include <libavdevice/avdevice.h>
    #include <libavutil/pixdesc.h>
    #include <libavutil/hwcontext.h>
    #include <libavutil/opt.h>
    #include <libavutil/avassert.h>
}

#include "post/queue.h"

class mp4Muxer
{
public:
    mp4Muxer( BlockingQueue<AVPacket*>* que, int alarFrameCnt,bool isVec = false);
    ~mp4Muxer();
    int *waitClose_;
    bool isVec_;
    bool IsInit = false;
    int alarFrameCnt;
    int waitFrameCnt = 999;
    bool isRecing = false;
    int64_t recfirFPts =0;
    int fps;
    AVFormatContext *pFormatCtx;//输入上下文 ps.这个输入上下文和类用到的队列应该由外部调用的地方释放
    AVFormatContext *oc = NULL;//输出上下文
    BlockingQueue<AVPacket*> *Rec_Frame_lists;
    std::thread* t_alarRecer = NULL;

    int muxer_init(std::string str);
    void muxer_deinit(std::time_t tv1);
    void muxer_deinit2();
    void av_packet_rescale_ts(AVPacket *pkt, AVRational src_tb, AVRational dst_tb);
    int WritePacket(AVPacket* packet);
    void startRec(AVFormatContext* input,std::string str,int *waitClose = NULL);
    void startRec2(AVFormatContext* input,std::string str,int *waitClose ,std::mutex &lock);
    void alarmRecer(std::string str);
    void alarmByVec(std::string str);
};



#endif



