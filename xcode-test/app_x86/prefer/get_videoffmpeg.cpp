#include "get_videoffmpeg.hpp"
#include "unistd.h"
#include <iostream>
#include "string.h"
#include "sys/time.h"
#include <thread>
#include<fcntl.h>
#include<sys/stat.h>

static int get_memory_by_pid(pid_t pid) {
    FILE* fd;
    char line[1024] = {0};
    char virtual_filename[32] = {0};
    char vmrss_name[32] = {0};
    int vmrss_num = 0;
    sprintf(virtual_filename, "/proc/%d/status", pid);
    fd = fopen(virtual_filename, "r");
    if(fd == NULL) {
        std::cout << "open " << virtual_filename << " failed" << std::endl;
        exit(1);
    }
    
    // VMRSS line is uncertain
    fread(line,1024,1,fd);
    std::string strs = line;
    int b_i = strs.find("VmRSS:");
    std::string mms = strs.substr(b_i,50);
    b_i = mms.find("\n");
    mms = mms.substr(0,b_i);
    std::cout<<"\n>>>>>>>>>>> "<<mms<<std::endl;

    fclose(fd);
    return vmrss_num;
}

/*
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
*/

cv::Mat avframe2cvmat(AVFrame *avframe, int w, int h) {
    struct timeval tv1,tv2,tv3,tv4;
    gettimeofday(&tv3,NULL);
    cv::Mat mat;
	// if (w <= 0) w = avframe->width;
	// if (h <= 0) h = avframe->height;
    //printf("w :%d,h:%d,pix:%s\n",avframe->width,avframe->height,av_get_pix_fmt_name((enum AVPixelFormat)avframe->format));

    if(w <= 0 || h <= 0)
        return mat;
	struct SwsContext *sws_ctx = NULL;
	sws_ctx = sws_getContext(w, h, (enum AVPixelFormat)avframe->format,
		w, h, AV_PIX_FMT_BGR24, SWS_BICUBIC, NULL, NULL, NULL);
	
	mat.create(cv::Size(w, h), CV_8UC3);
	AVFrame *bgr24frame = av_frame_alloc();
	bgr24frame->data[0] = (uint8_t *)mat.data;
	// avpicture_fill((AVPicture *)bgr24frame, bgr24frame->data[0], AV_PIX_FMT_BGR24, w, h);
    av_image_fill_arrays(bgr24frame->data, bgr24frame->linesize,bgr24frame->data[0], AV_PIX_FMT_BGR24, w, h, 1);
	sws_scale(sws_ctx,
		(const uint8_t* const*)avframe->data, avframe->linesize,
		0, h, // from cols=0,all rows trans
		bgr24frame->data, bgr24frame->linesize);
 
	av_free(bgr24frame);
	sws_freeContext(sws_ctx);
    gettimeofday(&tv4,NULL);
    std::cout<<"get frame use:"<<(tv4.tv_sec - tv3.tv_sec)*1000 + (tv4.tv_usec - tv3.tv_usec)/1000<<"->"<<std::endl;
	return mat;
}

// #ifdef _SelCard
int VideoDeal::hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type,int deviceId)
// #else
// int VideoDeal::hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type)
// #endif
{
    int err = 0;

    #ifdef _SelCard
    char device[128] = "";
    if (type == AV_HWDEVICE_TYPE_VAAPI) {
       snprintf(device, sizeof(device), "/dev/dri/renderD%d", 128 + deviceId);
    }else {
       snprintf(device, sizeof(device), "%d", deviceId);
    }

    if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type,device, NULL, 0)) < 0)
    #else
    if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type,NULL, NULL, 0)) < 0)
    #endif
    {
        fprintf(stderr, "Failed to create specified HW device.\n");
        return err;
    }
    ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
    return err;
}


bool needDumpPack(AVPacket *packet,int *dumpPackCnt,bool *isfinish)
{
    if((*dumpPackCnt))
    {
        if((packet->flags & AV_PKT_FLAG_KEY))
        {
            (*isfinish) = true;
            return false;
        }
    }

    av_packet_unref(packet);
    av_packet_free(&packet);
    (*dumpPackCnt)++;
    // std::cout<<"dump one frame"<<std::endl;
    return true;
}

void VideoDeal::getPacket()
{
    // std::cout<<"get packet start!"<<std::endl;
    AVPacket *packet = av_packet_alloc(); //分配一个packet
    avcodec_flush_buffers(pCodecCtx);
    end_of_stream = 1;
    bool firstIpacket = true;
    int dumpPackCnt = 0;
    int dumCnt = 0;
    while(b_getter && end_of_stream)
    {
        if (isFile)
        {
            usleep(1000000/fps);
            // usleep(10*1000);
        }

        // std::cout<<"in getPacket thread!"<<std::endl;
        if (av_read_frame(pFormatCtx, packet) < 0)
        {
            std::cout<<"end_of_stream"<<std::endl;
            av_packet_unref(packet);
            end_of_stream = 0;
            break;
        }

        if(isPause)
        {
            av_packet_unref(packet);
            usleep(10000);
            continue;
        }
        
        if(firstIpacket)
        {
            if(!(packet->flags & AV_PKT_FLAG_KEY))
            {
                av_packet_unref(packet);
                continue;
            }
            firstIpacket = false;
        }
        
        if(packet->size <= 0 || packet->stream_index != videoStream_best)
        {
            av_packet_unref(packet);
            continue;
        }

        //录像队列
        if(Rec_Frame_lists.get()!= nullptr)
        {
            AVPacket *copy;
            copy = av_packet_clone(packet);
            if(Rec_Frame_lists->Push(copy) != 0)
            {
                AVPacket *tmp;
                Rec_Frame_lists->Pop(tmp,100);
                av_packet_unref(tmp);
                av_packet_free(&tmp);
                if(Rec_Frame_lists->Push(copy) != 0)
                {
                    av_packet_unref(copy);
                    av_packet_free(&copy);
                }
            }
        }
        //解码队列
        if(packet_list.get()!= nullptr)
        {
            AVPacket *copy;
            copy = av_packet_clone(packet);
            if(packet_list->GetSize()>= packet_list->max_size_)
            {
                // std::cout<<"packetList is full"<<std::endl;
                int packetSize = 0;
                int dumpPackCnt = 0;
                bool finish = false;
                while(!finish)
                {
                    AVPacket *temp;
                    packetSize = packet_list->GetSize();
                    if(packetSize <= 0)
                    {
                        finish = true;
                        continue;
                    }
                    packet_list->ReadHead(temp,needDumpPack,&dumpPackCnt,&finish);
                    temp = NULL;
                }
                packetSize = packet_list->GetSize();
                // std::cout<<"packetList size:"<<packetSize<<",dump size:"<<dumpPackCnt<<","<<video_path<<std::endl;
            }

            if(packet_list->Push(copy) != 0) //fail
            {
                if(copy->flags&AV_PKT_FLAG_KEY)
                    std::cout<<"dump a I frame from tail"<<","<<video_path<<std::endl;
                else
                    std::cout<<"dump a bp frame from tail"<<","<<video_path<<std::endl;
                av_packet_unref(copy);
                av_packet_free(&copy);
            }
        }
        av_packet_unref(packet);
        usleep(5000);
    }
    av_packet_unref(packet);
    av_packet_free(&packet);
    #ifdef DecodeInfo
    std::cout<<"get packet out!"<<std::endl;
    #endif
}

void VideoDeal::getHwFrame()
{
    // std::cout<<"getHwFrame th start"<<std::endl;
    AVFrame *hwFrame;
    AVFrame *sw_Frame = av_frame_alloc();
    auto start = std::chrono::system_clock::now();
    auto begin = std::chrono::system_clock::now();
    while(b_getter && end_of_stream)
    {
        if(hwFrame_list.get()!= nullptr && hwFrame_list->GetSize()>0)
        {
            if(hwFrame_list->Pop(hwFrame,100) != 0)
            {
                usleep(10000);
                continue;
            }
                
            int nRet = 0;
            sw_Frame->width = hwFrame->width;
            sw_Frame->height = hwFrame->height;
            sw_Frame->format = AV_PIX_FMT_NV12;
            {
                if(testH != hwFrame->height)
                {
                    printf("-------------------------------------------------------------\n");
                    printf("[INFO] new H and W : %d %d ,%s\n",hwFrame->height,hwFrame->width,video_path.c_str());
                    printf("-------------------------------------------------------------\n");
                    
                    if(hwFrame->height < 200)
                    {
                        av_frame_unref(hwFrame);
                        av_frame_free(&hwFrame);
                        printf("[ERROR] drop this error,otherwise will cause av_hwframe_transfer_data() crash ,%s\n",video_path.c_str());
                        continue;
                    }
                    else
                    {
                        testH = hwFrame->height;
                        testW = hwFrame->width;
                    }
                }
            }
            // std::cout<<"hwFrame format:"<<hwFrame->format<<std::endl;
            if ((nRet = av_hwframe_transfer_data(sw_Frame, hwFrame, 0)) < 0) {
            // if ((nRet = av_hwframe_map(sw_Frame, hwFrame, 0)) < 0) {
                printf("av_hwframe_transfer_data fail %d\n",nRet);
                char str[200] = {0};
                av_strerror(nRet,str,200);
                printf("[Error] return:%s\n",str);
                av_frame_unref(hwFrame);
                av_frame_free(&hwFrame);
                av_frame_unref(sw_Frame);
                continue;
            }
            
            AVFrame* tmp = av_frame_clone(sw_Frame);
            if(frame_list.get() != nullptr && frame_list->Push(tmp) != 0)
            {
                av_frame_unref(tmp);
                av_frame_free(&tmp);
            }
            av_frame_unref(hwFrame);
            av_frame_free(&hwFrame);
            av_frame_unref(sw_Frame);

            begin = std::chrono::system_clock::now();
            useTimeArr.push_back((int)(std::chrono::duration_cast<std::chrono::milliseconds>(begin - start).count()));
            start = std::chrono::system_clock::now();

            if((useTimeArr.size() % 200) == 0)
            {   
                int preUse = int(std::accumulate(useTimeArr.begin(), useTimeArr.end(), 0) / useTimeArr.size());
                // std::cout<<"[DEC_INFO] aver :"<<preUse<<"ms, rtsp:"<<video_path<<"\n";
                useTimeArr.clear();
            }
        }
        else
        {
            usleep(5000);
            // continue;
        }
        
    }
    av_frame_unref(sw_Frame);
    av_frame_free(&sw_Frame);
    // av_frame_unref(hwFrame);
    // av_frame_free(&hwFrame);
    #ifdef DecodeInfo
    std::cout<<"getHwFrame th out"<<std::endl;
    #endif
}

void VideoDeal::getter()
{
    int index = 1;
    int ret = 0;
    AVFrame *pFrameRGB;
    pFrameRGB = av_frame_alloc();
    AVFrame *sw_Frame = av_frame_alloc();
    //=========================== 分配AVPacket结构体 ===============================//
    // AVPacket *packet = av_packet_alloc(); //分配一个packet
    // avcodec_flush_buffers(pCodecCtx);
    //===========================  读取视频信息 ===============================//
    int got_frame = 0;

    std::thread t_getPacketer =  std::thread(&VideoDeal::getPacket,this);
    std::thread t_getHwFrame =  std::thread(&VideoDeal::getHwFrame,this);
    std::thread resize_th(&VideoDeal::resize_frame,this);

    struct timeval tv1,tv2,tv3,tv4;
    int nRet = 0; 
    while (b_getter && end_of_stream)
    {   
        if(cams_count>16)
            usleep(cams_count*20*1000);//！！！路数多的时候要抽帧
        AVPacket *packet = 0;
        if (packet_list->GetSize() > 0 && packet_list->Pop(packet,100) == 0)
        {
            got_frame = 0;    
            nRet = avcodec_send_packet(pCodecCtx, packet);
            av_packet_unref(packet);
            av_packet_free(&packet);
            packet = NULL;
            AVFrame *hw_frame = av_frame_alloc();;
            nRet = avcodec_receive_frame(pCodecCtx, hw_frame);
            if (nRet != 0)
            {
                av_frame_unref(hw_frame);
                av_frame_free(&hw_frame);
                char str[200] = {0};
                av_strerror(nRet,str,200);
                // printf("%s \n",str);
            }

            if (nRet == 0)
            {
                got_frame = 1;
                if(hw_decode)
                {
                    AVFrame* tmp = av_frame_clone(hw_frame);
                    av_frame_unref(hw_frame);
                    av_frame_free(&hw_frame);
                    if(hwFrame_list.get() != nullptr && hwFrame_list->Push(tmp) != 0)
                    {
                        av_frame_unref(tmp);
                        av_frame_free(&tmp);
                    }
                    gettimeofday(&tv4,NULL);
                    // std::cout<<"decode use:"<<(tv4.tv_sec - tv3.tv_sec)*1000 + (tv4.tv_usec - tv3.tv_usec)/1000<<","<<video_path<<std::endl;
                    gettimeofday(&tv3,NULL);

                }
                else{
                    AVFrame* tmp = av_frame_clone(hw_frame);
                    av_frame_unref(hw_frame);
                    if(frame_list.get() != nullptr && frame_list->Push(tmp) != 0)
                    {
                        av_frame_unref(tmp);
                        av_frame_free(&tmp);
                    }
                }
                
                
                if(packet)
                {
                    av_packet_unref(packet);
                    av_packet_free(&packet);
                    packet = NULL;
                }
                av_frame_unref(hw_frame);
                av_frame_free(&hw_frame);
                av_frame_unref(sw_Frame);
                continue;
            }

        }
        else
            usleep(10000);

        if(packet)
        {
            av_packet_unref(packet);
            av_packet_free(&packet);
            packet = NULL;
        }
        av_frame_unref(pFrameRGB);
        av_frame_unref(sw_Frame);

    }
    if(packet_list->GetSize()>0 && isFile)
        sleep(3);
    #ifdef DecodeInfo
    std::cout<<"decode th out begin!"<<std::endl;
    #endif
    b_getter = false;
    av_frame_unref(sw_Frame);
    av_frame_unref(pFrameRGB);
    av_frame_free(&pFrameRGB);
    av_frame_free(&sw_Frame);
    av_free(pFrameRGB);
    av_free(sw_Frame);

    if(resize_th.joinable())
        resize_th.join();
    resize_in_list->Clear();
    resize_out_list->Clear();
    resize_in_list.reset();
    resize_out_list.reset();

    if (t_getPacketer.joinable())
    {
        t_getPacketer.join();
    }

    if (t_getHwFrame.joinable())
    {
        t_getHwFrame.join();
    }
    
    if(packet_list.get()!= nullptr)
    {
        AVPacket* tmp = NULL;
        while(packet_list->GetSize() > 0)
        {
            packet_list->Pop(tmp,100);
            av_packet_unref(tmp);
            av_packet_free(&tmp);
            tmp = NULL;
        }
        packet_list->Clear();
        #ifdef DecodeInfo
        std::cout<<"packet_list clear"<<"\n";
        #endif
    }
    packet_list.reset();

    if(hwFrame_list.get()!= nullptr)
    {
        AVFrame* tmp = NULL;
        while(hwFrame_list->GetSize() > 0)
        {
            hwFrame_list->Pop(tmp,100);
            av_frame_unref(tmp);
            av_frame_free(&tmp);
            tmp = NULL;
        }
        hwFrame_list->Clear();
        #ifdef DecodeInfo
        std::cout<<"hwFrame_list clear"<<"\n";
        #endif
    }
    hwFrame_list.reset();

    if(Rec_Frame_lists.get()!= nullptr)
    {
        AVPacket* tmp = NULL;
        while(Rec_Frame_lists->GetSize() > 0)
        {
            Rec_Frame_lists->Pop(tmp,100);
            av_packet_unref(tmp);
            av_packet_free(&tmp);
            tmp = NULL;
        }
        Rec_Frame_lists->Clear();
        #ifdef DecodeInfo
        std::cout<<"Rec_Frame_lists clear"<<"\n";
        #endif
    }
    Rec_Frame_lists.reset();

    if(frame_list.get()!= nullptr)
    {
        AVFrame* tmp = NULL;
        while(frame_list->GetSize() > 0)
        {
            //std::cout<<"free one frame!"<<std::endl;
            frame_list->Pop(tmp,100);
            av_frame_unref(tmp);
            av_frame_free(&tmp);
            tmp = NULL;
        }
        frame_list->Clear();
        #ifdef DecodeInfo
        std::cout<<"frame_list clear"<<"\n";
        #endif
    }
    frame_list.reset();


    av_buffer_unref(&hw_device_ctx);
    av_buffer_unref(&pCodecCtx->hw_device_ctx);

    avcodec_flush_buffers(pCodecCtx);
    // avcodec_close(pCodecCtx); //关闭视频解码器
    avcodec_free_context(&pCodecCtx);

    avformat_flush(pFormatCtx);
    try
    {
        avformat_close_input(&pFormatCtx);
        pFormatCtx = NULL;
        //avformat_free_context(pFormatCtx);
    } //关闭输入文件
    catch (...)
    {
        std::cout<<"input close error"<<std::endl;
    }

    pCodecCtx = nullptr;

    
    #ifdef DecodeInfo
    std::cout<<"decode thread is down!:"<<b_getter<<status<<end_of_stream<<std::endl;
    #endif
    status = false;
}

void VideoDeal::resize_frame()
{
    while(b_getter && end_of_stream)
    {
        cv::Mat img;
        if(resize_in_list->GetSize() <= 0)
        {
            usleep(50000);
            continue;
        }
        if(resize_in_list->Pop(img,100) == 0)
        {
            if(isReSize)
                sw_resize(img);
            resize_out_list->Push(img);
        }
        else
            usleep(10000);
    }
    #ifdef DecodeInfo
    std::cout<<"resize_frame out"<<"\n";
    #endif
}

int VideoDeal::main_ffmpeg(const char *file_path)
{
    av_log_set_level(AV_LOG_PANIC);     
    // av_log_set_level(AV_LOG_ERROR);
    // av_log_set_level(AV_LOG_QUIET);
    av_register_all();       //初始化FFMPEG  调用了这个才能正常适用编码器和解码器
    avformat_network_init(); // 网络初始化
    g_file_path = file_path;
    enum AVHWDeviceType type;
    AVCodec *pCodec;

    std::string file_path_str = file_path;

    if (file_path_str.find(".mp4") != std::string::npos)
    {
        if((file_path_str.find("rtsp://") != std::string::npos) || (file_path_str.find("rtmp://") != std::string::npos))
        {
            isFile = false;
        }
        else{
            if(access(file_path, 0) == -1)
            {
                std::cout<<file_path_str<<":video path not exist!"<<"\n";
                return -1;
            }
            else
                isFile = true;
        }
    }

    //=========================== 创建AVFormatContext结构体 ===============================//
    //分配一个AVFormatContext，FFMPEG所有的操作都要通过这个AVFormatContext来进行
    pFormatCtx = avformat_alloc_context();
    //==================================== 打开文件 ======================================//
    // char *file_path = argv[1];//这里必须使用左斜杠
    AVDictionary *avDictionary = nullptr;
    if(!isFile)
        av_dict_set(&avDictionary, "fflags", "nobuffer", 0);   //无缓存，解码时有效
    // av_dict_set(&avDictionary, "probesize", "1048576", 0); // 2048
    // av_dict_set(&avDictionary, "max_analyze_duration", "5", 0);
    av_dict_set(&avDictionary, "stimeout", "10000000", 0);  //设置超时断开连接时间us
    av_dict_set(&avDictionary, "rw_timeout", "10000000", 0); //ms   
    av_dict_set(&avDictionary, "buffer_size", "409600", 0);
    av_dict_set(&avDictionary, "rtsp_transport", "tcp", 0); // 设置使用tcp传输,因udp在1080p下会丢包导致花屏
    int ret = avformat_open_input(&pFormatCtx, file_path, nullptr, &avDictionary);
    av_dict_free(&avDictionary);
    if (ret != 0)
    {
        char str[200] = {0};
        av_strerror(ret,str,200);
        printf("av_strerror:%s\n",str);
        printf("error:%s",file_path);
        avformat_close_input(&pFormatCtx);   
        return -1;
    }

    type = av_hwdevice_find_type_by_name("cuda");

    if (type == AV_HWDEVICE_TYPE_NONE)
    {
        fprintf(stderr, "Device type %s is not supported.\n", "cuda");
        fprintf(stderr, "Available device types:");
        while ((type = av_hwdevice_iterate_types(type)) != AV_HWDEVICE_TYPE_NONE)
            fprintf(stderr, " %s", av_hwdevice_get_type_name(type));
        fprintf(stderr, "\n");
        return -1;
    }

    //=================================== 获取视频流信息 ===================================//
    if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
    {
        printf("Could't find stream infomation.");
        return -1;
    }
    videoStream_best = av_find_best_stream(pFormatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, &pCodec, 0);
    if (videoStream_best < 0)
    {
        printf("Didn't find a video stream.");
        return -1;
    }
    //=================================  查找解码器 ===================================//
    for (int i = 0;; i++)
    {
        const AVCodecHWConfig *config = avcodec_get_hw_config(pCodec, i);
        if (!config)
        {
            fprintf(stderr, "Decoder %s does not support device type %s.\n",
                    pCodec->name, av_hwdevice_get_type_name(type));
            return -1;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            config->device_type == type)
        {
            hw_pix_fmt = config->pix_fmt;
            break;
        }
    }
    if (!(pCodecCtx = avcodec_alloc_context3(pCodec)))
        return AVERROR(ENOMEM);

    pCodecCtx->extra_hw_frames = 8;//针对No decoder surfaces left的问题，貌似是遇到了高分辨率的图片申请不到内存，参数越大

    width = pFormatCtx->streams[videoStream_best]->codecpar->width;
    height = pFormatCtx->streams[videoStream_best]->codecpar->height;
    fps = av_q2d(pFormatCtx->streams[videoStream_best]->r_frame_rate);
    // #ifdef DecodeInfo
    // std::cout<<"rtsp fps:"<<fps<<","<<file_path<<std::endl;
    // #endif
    if(fps <= 0 || fps > 60)
        fps = 25;
    alarFrameCnt = fps * 10;

    frame_list.reset(new BlockingQueue<AVFrame*>(3));
    Rec_Frame_lists.reset(new BlockingQueue<AVPacket*>((int)(alarFrameCnt)));
    resize_in_list.reset(new BlockingQueue<cv::Mat>(3));
    resize_out_list.reset(new BlockingQueue<cv::Mat>(3));
    hwFrame_list.reset(new BlockingQueue<AVFrame*>(3));
    packet_list.reset(new BlockingQueue<AVPacket*>(50));

    if(avcodec_parameters_to_context(pCodecCtx, pFormatCtx->streams[videoStream_best]->codecpar) < 0)
        return -1;

    // std::cout<<"rtsp fps:"<<fps<<" alarFrameCnt:"<<alarFrameCnt<<std::endl;
    
    #ifdef _SelCard
    if (hw_decoder_init(pCodecCtx, type,_deviceId) < 0)
    #else
    if (hw_decoder_init(pCodecCtx, type) < 0)
    #endif
        return -1;

    //================================  打开解码器 ===================================//
    if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) // 具体采用什么解码器ffmpeg经过封装  我们无须知道
    {
        printf("Could not open codec.");
        return -1;
    }

    b_getter = true;
    t_getter = new std::thread(&VideoDeal::getter, this);
    return 0;
}

// #ifdef _SelCard
VideoDeal::VideoDeal(std::string videoPath,int deviceId):video_path(videoPath),_deviceId(deviceId)
// #else
// VideoDeal::VideoDeal(std::string videoPath):video_path(videoPath)
// #endif
{
    // return ;
    //av_log_set_level(AV_LOG_QUIET);
    
    isPause = false;
    b_getter = false;
    #ifdef DECNOTRESIZE
    isReSize = false;
    #else
    isReSize = true;
    #endif
    // printf("video path : %s\n", videoPath.c_str());
    int ret;
    if ((videoPath.find("rtsp") != std::string::npos))
    {
        // printf("video path not exist!");
        status = true;
        // return;
    }
    else if ((videoPath.find(".mp4") != std::string::npos))
    {
        if (access(videoPath.c_str(), 0) == -1)
        {
            status = false;
            return;
        }
    }
    status = true;

    ret = main_ffmpeg(videoPath.c_str()); // init

    if (ret != 0)
    {
        // printf("\nvideo open error\n");
        status = false;
    }
    else
    {
        width = (int)pCodecCtx->width;
        height = (int)pCodecCtx->height;
        bvideoRead = true;
        status = true;
    }
    sleep(1);//确保线程启动
}

VideoDeal::~VideoDeal()
{
    // std::cout<<"~VideoDeal() begin:"<<video_path<<std::endl;  

    b_getter = false;

    if (t_getter != nullptr && t_getter->joinable())
    {
        t_getter->join();
        delete t_getter;
    } 
    #ifdef DecodeInfo
    std::cout<<"~VideoDeal() end:"<<video_path<<std::endl;    
    #endif

}

cv::Mat ConvertMat(AVFrame* pFrame, int width, int height)
{   //nv12（yuv420sp）存储格式为yyyuvuvuv,在AVFrame中data[0]存yyy,data[1]存uvuvuv
    //(yuv420p)则为yyy uuu vvv 分别对应data[0],[1],[2]
    //rgb32则为data[0]
    struct timeval tv1,tv2,tv3,tv4;
    static int wtime = 0;
    // gettimeofday(&tv3,NULL);
    cv::Mat bgr= cv::Mat::zeros( height, width, CV_8UC3);
    cv::Mat tmp_img = cv::Mat::zeros( height*3/2, width, CV_8UC1); 
    // printf("\nbegin ConvertMat,tmp_img size:%d\n",tmp_img.size());   
    memcpy( tmp_img.data, pFrame->data[0], width*height);
    memcpy( tmp_img.data + width*height, pFrame->data[1], width*height/2);
    
    //memcpy( tmp_img.data + width*height*5/4, pFrame->data[2], width*height/4 );
    // wtime++;
    // if(wtime > 200)
    // {
    //     std::cout<<"format:"<<pFrame->format<<std::endl;
    //     cv::imwrite("changed.jpg", tmp_img);
    // }
    cv::cvtColor(tmp_img, bgr, cv::COLOR_YUV2BGR_NV12);

    av_frame_unref(pFrame);
    // av_frame_free(&pFrame);
    // AVPixelFormat
    tmp_img.release();
    // gettimeofday(&tv4,NULL);
    // std::cout<<"get frame use:"<<(tv4.tv_sec - tv3.tv_sec)*1000 + (tv4.tv_usec - tv3.tv_usec)/1000<<"->"<<std::endl;
    return bgr;
}

int VideoDeal::getframe(cv::Mat &ansMat)
{
    struct timeval tv1,tv2,tv3,tv4;
    if(b_getter)
    {
        AVFrame* frame = NULL;
        if(frame_list.get()!= nullptr && frame_list->GetSize() == 0)
            usleep(10000);
        
        if(frame_list.get()!= nullptr && frame_list->GetSize()>=1)
        {
            cv::Mat mat;
            if(frame_list->Pop(frame,100) != 0)
            {
                usleep(10000);
                return 1;
            }
            gettimeofday(&tv1,NULL);
            mat = ConvertMat(frame,frame->width,frame->height);
            gettimeofday(&tv2,NULL);
            // std::cout<<"toMat:"<<(tv2.tv_sec - tv1.tv_sec)*1000 + (tv2.tv_usec - tv1.tv_usec)/1000<<"\n";
            av_frame_unref(frame);
            av_frame_free(&frame);
            frame = NULL;
            if(mat.empty())
            {
                std::cout<<"---------------------mat.empty()! ---------------------------"<<std::endl;
                return 1; //img may error
            }
            if(resize_in_list.get()!= nullptr)
            {
                resize_in_list->Push(mat);
                if(resize_out_list.get()!= nullptr && resize_out_list->GetSize()>0)
                {
                    if(resize_out_list.get()!= nullptr && resize_out_list->Pop(ansMat,100) == 0)
                        return 0;
                    else
                    {
                        usleep(10000);
                        return 1;
                    }
                }
            }
        }
        //std::cout<<"---------------------list is empty! ---------------------------"<<std::endl;
        return 1; //list is empty
    }
    std::cout<<"---------------------decode thread is dead! ----------------------:"<<b_getter<<std::endl;
    return -2; //thread is dead
}

AVPacket* VideoDeal::getPacketFromQue()
{
    AVPacket *packet = NULL;
    if(Rec_Frame_lists.get() != nullptr && Rec_Frame_lists->GetSize()>0)
        Rec_Frame_lists->Pop(packet,100);
    return packet;
}


#if 1
int VideoDeal::getNewestFrame(cv::Mat &ansMat, int &index) {
    struct timeval tv1,tv2,tv3,tv4;
    if(b_getter)
    {
        AVFrame* frame = NULL;
        if(frame_list.get()!= nullptr && frame_list->GetSize() == 0)
            usleep(10000);

        if(frame_list.get()!= nullptr && frame_list->GetSize()>=1)
        {
            cv::Mat mat;
            if(frame_list->Pop(frame,100) != 0)
            {
                mMatLock.read_lock();
                ansMat = m_cv_mat;
                index = mIndex;
                mMatLock.read_release();
                usleep(10000);
                return 1;
            }
            gettimeofday(&tv1,NULL);
            mat = ConvertMat(frame,frame->width,frame->height);
            gettimeofday(&tv2,NULL);
            // std::cout<<"toMat:"<<(tv2.tv_sec - tv1.tv_sec)*1000 + (tv2.tv_usec - tv1.tv_usec)/1000<<"\n";
            av_frame_unref(frame);
            av_frame_free(&frame);
            frame = NULL;
            if(mat.empty())
            {
                mMatLock.read_lock();
                ansMat = m_cv_mat;
                index = mIndex;
                mMatLock.read_release();
                std::cout<<"---------------------mat.empty()! ---------------------------"<<std::endl;
                return 1; //img may error
            }
            if(resize_in_list.get()!= nullptr)
            {
                resize_in_list->Push(mat);
                if(resize_out_list.get()!= nullptr && resize_out_list->GetSize()>0)
                {
                    if(resize_out_list.get()!= nullptr && resize_out_list->Pop(ansMat,100) == 0)
                    {
                        mMatLock.write_lock();
                        mIndex++;
                        index = mIndex;
                        m_cv_mat = ansMat;
                        mMatLock.write_release();
                        return 0;
                    }
                    else
                    {
                        mMatLock.read_lock();
                        ansMat = m_cv_mat;
                        index = mIndex;
                        mMatLock.read_release();
                        usleep(10000);
                        return 1;
                    }
                }
            }
        }
        mMatLock.read_lock();
        ansMat = m_cv_mat;
        index = mIndex;
        mMatLock.read_release();
        //std::cout<<"---------------------list is empty! ---------------------------"<<std::endl;
        return 1; //list is empty
    }
    
    // std::cout<<"---------------------decode thread is dead! ----------------------:"<<b_getter<<std::endl;
    return -2; //thread is dead
}

#endif