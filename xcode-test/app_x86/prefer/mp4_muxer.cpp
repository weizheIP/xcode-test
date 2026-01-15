

#include "mp4_muxer.hpp"


int mp4Muxer::muxer_init(std::string str)
{
    int ret  = avformat_alloc_output_context2(&oc, nullptr, "mp4", str.c_str());
    int videoStream_best = -1;
    AVStream * stream = nullptr;
	if(ret < 0)
	{
		av_log(NULL, AV_LOG_ERROR, "open output context failed\n");
		goto Error;
	}

	ret = avio_open2(&oc->pb, str.c_str(), AVIO_FLAG_WRITE,nullptr, nullptr);
	if(ret < 0)
	{
		av_log(NULL, AV_LOG_ERROR, "open avio failed");
		goto Error;
	}

	// for(int i = 0; i < pFormatCtx->nb_streams; i++)
	// {
        videoStream_best = av_find_best_stream(pFormatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        stream = avformat_new_stream(oc, nullptr); //给out_AVFormatContext->stream分配空间
		avcodec_parameters_copy(stream->codecpar, pFormatCtx->streams[videoStream_best]->codecpar);//拷贝编解码参数
		if(ret < 0)
		{
			av_log(NULL, AV_LOG_ERROR, "copy coddec context failed");
			goto Error;
		}
	//}

	ret = avformat_write_header(oc, nullptr);
	if(ret < 0)
	{

		av_log(NULL, AV_LOG_ERROR, "format write header failed");
		goto Error;
	}

	av_log(NULL, AV_LOG_FATAL, " Open output file success %s\n",str.c_str());
	return ret ;
Error:
	if(oc)
	{
		for(int i = 0; i < oc->nb_streams; i++)
		{
			// avcodec_close(oc->streams[i]->codec);
		}
		avformat_close_input(&oc);
	}
	return ret ;
}

void mp4Muxer::muxer_deinit(std::time_t tv1)
{
    std::time_t tv2;
    if(oc != nullptr)
	{
        av_write_trailer(oc);
		for(int i = 0 ; i < oc->nb_streams; i++)
		{
			// AVCodecContext *codecContext = oc->streams[i]->codec;
			// avcodec_close(codecContext);
		}
		// avformat_free_context(oc);
        avformat_close_input(&oc);
	}

    tv2 = std::time(0);
    while(tv2 - tv1 <= 60)//分钟只能录像一次
    {
        tv2 = std::time(0);
        usleep(100*1000);
    }
    waitFrameCnt = 999;
    recfirFPts = 0;
    isRecing = false;
    // printf("rec deinit !\n");
}

void mp4Muxer::muxer_deinit2()
{
    std::time_t tv2;
    if(oc != nullptr)
	{
        av_write_trailer(oc);
		for(int i = 0 ; i < oc->nb_streams; i++)
		{
			// AVCodecContext *codecContext = oc->streams[i]->codec;
			// avcodec_close(codecContext);
		}
		// avformat_free_context(oc);
        avformat_close_input(&oc);
	}

    tv2 = std::time(0);
    
    waitFrameCnt = 999;
    recfirFPts = 0;
    isRecing = false;
    // printf("rec deinit !\n");
}

void mp4Muxer::av_packet_rescale_ts(AVPacket *pkt, AVRational src_tb, AVRational dst_tb)
{
    if (pkt->pts != AV_NOPTS_VALUE)
		pkt->pts = av_rescale_q(pkt->pts, src_tb, dst_tb) - recfirFPts; //av_rescale_q(a,b,c)是用来把时间戳从一个时基调整到另外一个时基时候用的函数。
	if (pkt->dts != AV_NOPTS_VALUE)
		pkt->dts = av_rescale_q(pkt->dts, src_tb, dst_tb) - recfirFPts;
    if(recfirFPts == 0)
    {
       recfirFPts = pkt->pts;
       pkt->dts = 0;
       pkt->pts = 0; //我会从i帧开始，正常来说i帧的pts和dts是一样的，非i帧只要没有
    }
	if (pkt->duration > 0)
		pkt->duration = av_rescale_q(pkt->duration, src_tb, dst_tb);
}

int mp4Muxer::WritePacket(AVPacket* packet)
{ 
    auto inputStream = pFormatCtx->streams[packet->stream_index];
	auto outputStream = oc->streams[packet->stream_index];
	av_packet_rescale_ts(packet,inputStream->time_base,outputStream->time_base);
    // printf("dts:%d_%d_%d\n",packet->dts,packet->pts,packet->flags&AV_PKT_FLAG_KEY);
	return av_interleaved_write_frame(oc, packet);
}

void mp4Muxer::alarmRecer(std::string str){
    if(pFormatCtx == NULL || Rec_Frame_lists == NULL)
    {
        std::cout<<"rec is not init"<<"\n";
        return;
    }
    if(isRecing)
    {
        std::cout<<"[ERROR] rec is not stop ,please finish last rec!"<<"\n";
        
        return;
    }
    isRecing = true;
    if(str.empty())
    {
       std::cout<<"recFileName is empty !"<<"\n";
        return ;
    }
   
    std::time_t tv1,tv2;
    tv1 = std::time(0);
    int dumpCnt = alarFrameCnt/2;
    int wirteCnt = alarFrameCnt;
    int ret = muxer_init(str);
    if(0 != ret)
    {
        std::cout<<"-------encode_init fail!"<<"\n";
        goto Error;
    }
    while(0 <= dumpCnt --)//扔一半旧数据
    {
        AVPacket *packet;
        if(Rec_Frame_lists->Pop(packet,100)==0)
        {
            av_packet_unref(packet);
            av_packet_free(&packet);
            av_free(packet);
        }
    }
    while(0 < wirteCnt)
    {
        tv2 = std::time(0);
        if(tv2 - tv1 >= 10)//防止录像时间过长
            break;
        AVPacket *packet;
        if(Rec_Frame_lists->GetSize() <= 0 || 0 != Rec_Frame_lists->Pop(packet,100))
        {
            // std::cout<<"rec que is empty or error !"<<"\n";
            usleep(30*1000);
            continue;
        }
        else
        {
            // printf("dts:%d_%d\n",packet->dts,packet->flags&AV_PKT_FLAG_KEY);
            if(recfirFPts == 0 && !(packet->flags&AV_PKT_FLAG_KEY))//获取首帧为I帧
            {
                av_packet_unref(packet);
                av_packet_free(&packet);
                av_free(packet);
                continue;
            } 
            else
            {
                wirteCnt--;
                WritePacket(packet);
                av_packet_unref(packet);
                av_packet_free(&packet);
                av_free(packet);
                // std::cout<<"write one frame succss !"<<"\n";
            }
        }
    }
    
Error:
    muxer_deinit(tv1);
    return ;
}

void mp4Muxer::alarmByVec(std::string str)
{
    if(pFormatCtx == NULL || Rec_Frame_lists == NULL)
    {
        std::cout<<"rec is not init"<<"\n";
        return;
    }
    if(isRecing)
    {
        std::cout<<"[ERROR] rec is not stop ,please finish last rec!"<<"\n";
        
        return;
    }
    isRecing = true;
    if(str.empty())
    {
       std::cout<<"recFileName is empty !"<<"\n";
        return ;
    }
    std::time_t tv1,tv2;
    tv1 = std::time(0);
    int wirteCnt = alarFrameCnt;
    int ret = muxer_init(str);
    if(0 != ret)
    {
        std::cout<<"-------encode_init fail!"<<"\n";
        goto Error;
    }
    while(Rec_Frame_lists->GetSize()>0)
    {
        tv2 = std::time(0);
        if(tv2 - tv1 >= 10)//防止录像时间过长
            break;

        AVPacket *packet;
        if(Rec_Frame_lists->GetSize() > 0 && 0 == Rec_Frame_lists->Pop(packet,100))
        {
            // std::cout<<"rec que is empty or error !"<<"\n";
            if(recfirFPts == 0 && !(packet->flags&AV_PKT_FLAG_KEY))//获取首帧为I帧
            {
                av_packet_unref(packet);
                av_packet_free(&packet);
                av_free(packet);
                continue;
            } 
            else
            {
                wirteCnt--;
                WritePacket(packet);
                av_packet_unref(packet);
                av_packet_free(&packet);
                av_free(packet);
                // std::cout<<"write one frame succss !"<<"\n";
            }
        }
    }
Error:
    muxer_deinit2();
    return ;
}
void mp4Muxer::startRec(AVFormatContext* input,std::string str,int *waitClose)
{
    if(t_alarRecer != NULL)
    {
        if(t_alarRecer->joinable())
            t_alarRecer->join();
        delete t_alarRecer;
        t_alarRecer = NULL;
    }
    
    if(isRecing)
    {
        printf("is recording thread fail!\n");
        return;
    }
    if(input->streams == NULL)
    {
        printf("streams is NULL fail!\n");
        return;
    }

    pFormatCtx = input;
    // t_alarRecer = new std::thread(&mp4Muxer::alarmRecer,this,str);
    if(isVec_)
    {
        // t_alarRecer = new std::thread(&mp4Muxer::alarmByVec,this,str);
        // t_alarRecer->join();
        // delete t_alarRecer;
        // t_alarRecer = NULL;

        alarmByVec(str);
        
        while(Rec_Frame_lists->GetSize())
        {
            AVPacket *packet = NULL;
            Rec_Frame_lists->Pop(packet,10);
            if(packet)
            {
                av_packet_unref(packet);
                av_packet_free(&packet);
                av_free(packet);
            }
        }
        delete Rec_Frame_lists;
    }
    else
        t_alarRecer = new std::thread(&mp4Muxer::alarmRecer,this,str);
    // printf("alarRecer thread start !\n");
}


void mp4Muxer::startRec2(AVFormatContext* input,std::string str,int *waitClose,std::mutex &lock)
{
    if(t_alarRecer != NULL)
    {
        if(t_alarRecer->joinable())
            t_alarRecer->join();
        delete t_alarRecer;
        t_alarRecer = NULL;
    }
    
    if(isRecing)
    {
        printf("is recording thread fail!\n");
        return;
    }
    if(input->streams == NULL)
    {
        printf("streams is NULL fail!\n");
        return;
    }

    pFormatCtx = input;
    // t_alarRecer = new std::thread(&mp4Muxer::alarmRecer,this,str);
    if(isVec_)
    {
        {
            std::lock_guard<std::mutex> lk(lock);  // 加锁
            *waitClose = *waitClose + 1;
        }  
        // t_alarRecer = new std::thread(&mp4Muxer::alarmByVec,this,str);
        // t_alarRecer->join();
        // delete t_alarRecer;
        // t_alarRecer = NULL;
        alarmByVec(str);
        while(Rec_Frame_lists->GetSize())
        {
            AVPacket *packet = NULL;
            Rec_Frame_lists->Pop(packet,10);
            if(packet)
            {
                av_packet_unref(packet);
                av_packet_free(&packet);
                av_free(packet);
            }
        }
        delete Rec_Frame_lists;
        {
            std::lock_guard<std::mutex> lk(lock);  // 加锁
            *waitClose = *waitClose - 1;
        }  
    }
    else
        t_alarRecer = new std::thread(&mp4Muxer::alarmRecer,this,str);
    // printf("alarRecer thread start !\n");
}

mp4Muxer::mp4Muxer(BlockingQueue<AVPacket*>* que, int alarFrameCnt,bool isVec):
isRecing(false),
waitClose_(NULL)
{
    isVec_ = isVec;
    
    Rec_Frame_lists = que;
    
    this->alarFrameCnt = alarFrameCnt;
}

mp4Muxer::~mp4Muxer()
{
    if(t_alarRecer != NULL)
    {
        if(t_alarRecer->joinable())
            t_alarRecer->join();
        delete t_alarRecer;
        t_alarRecer = NULL;
    }
    // std::cout<<"rec th out"<<"\n";
}