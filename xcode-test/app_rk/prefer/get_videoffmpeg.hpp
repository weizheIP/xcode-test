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
#include <atomic>
#include <mutex>
#include "post/queue.h"

#include "common/alarmVideo/alarm_video.h"


#define __THREAD__

extern "C"
{
    #include "libavcodec/avcodec.h"
    #include "libavformat/avformat.h"
    #include "libswscale/swscale.h"
    #include "libavdevice/avdevice.h"
    #include "libavutil/pixfmt.h"
    #include "libavutil/imgutils.h"
    #include "libavutil/opt.h"
    // #include "mpp_frame.h"
    // #include "mpp_buffer_impl.h"
    // #include "mpp_frame_impl.h"
}
// struct decodeFrame
// {
//   MppFrame frame;  
// };

static void sw_resize(cv::Mat &tempImg1)
{
    #ifdef Model640
    int size = 640;
    #else
    int size = 1280;
    #endif
    if(tempImg1.cols != size && !tempImg1.empty())
    {   cv::Mat dstimg;
        int w = size, h = 0;
        h = tempImg1.rows * size /tempImg1.cols;
        //std::cout<<tempImg1.cols<<tempImg1.rows<<"-"<<dstw<<":"<<dsth<<std::endl;
        if(h%2)
            h++;
        cv::resize(tempImg1, dstimg, cv::Size(w, h), cv::INTER_AREA);
        tempImg1 = dstimg.clone();
        dstimg.release();
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
    #ifdef USBCAM
    bool isUsbCam = false;
    void getYuvFrame();
    std::thread* yuvThPtr = nullptr;
    BlockingQueue<cv::Mat> *usbCamQue;
    #endif

    int countOfEof;
    time_t TimeNow;
    VideoDeal(){;}
    VideoDeal(std::string videoPath, int gpu_device = 0);

    int getframe(cv::Mat &mat);
    AVPacket* getPacketFromQue();
    //录像
    int encode_init(const char *outfile);
    void encode_deinit();
//    void av_packet_rescale_ts(AVPacket *pkt, AVRational src_tb, AVRational dst_tb);
    int WritePacket(AVPacket* packet);
    void startRec(char *filename);
    
    std::mutex release_vid;

    ~VideoDeal();
    //输入
    AVFormatContext *pFormatCtx;
    AVCodecContext* pCodecCtx;
    bool hw_decode = true;

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
    bool isReSize;

	//生成报警视频参数
	std::mutex rec_frame_mtx;
	std::condition_variable rec_alarm_cv;	
	std::map<std::string, bool> cur_pkt_update_status;	
	std::mutex rec_alarm_mtx;
	bool alarm_stop;
	AVPacket *rec_pkt = nullptr;
	std::vector<std::string> cur_pkt_ids;

    //2、输出

    AVFormatContext *oc = NULL;
    int videoStream_best;

    bool bvideoSave = false;
    bool bvideoRead = false;

    volatile bool status;

    int end_of_stream = 1;
    // std::atomic_int end_of_stream{1};

	int write_file(std::shared_ptr<T_AlarmInfo> info);
	void alarmVideo();

    void getter();
    void alarmRecer(char *filename);
    
    std::shared_ptr<BlockingQueue<AVPacket*>> Rec_Frame_lists;
    std::shared_ptr<BlockingQueue<AVPacket*>> packet_list;

    std::shared_ptr<BlockingQueue<AVFrame*>> frame_list;
    std::shared_ptr<BlockingQueue<AVFrame*>> hwFrame_list;
    std::shared_ptr<BlockingQueue<cv::Mat>> resize_in_list;
    std::shared_ptr<BlockingQueue<cv::Mat>> resize_out_list;
    
    void getPacket();
    void getHwFrame();
    void resize_frame();
    
    cv::Mat g_img_decode;
    std::mutex mlock;
    std::mutex tlock;
    bool b_getter;
    std::thread *t_getter = nullptr;
    std::thread *t_alarRecer = nullptr;

    
    int main_ffmpeg(const char * file_path);
    
};

#endif

