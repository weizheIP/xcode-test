#include "get_videoffmpeg.hpp"
#include "unistd.h"
#include <iostream>
#include "string.h"
#include "sys/time.h"
#include <thread>
#include <fcntl.h>
#include <sys/stat.h>



//static void sw_resize(cv::Mat &tempImg1)
//{
//    #ifdef Model640
//    int size = 640;
//    #else
//    int size = 1280;
//    #endif
//    if(tempImg1.cols != size && !tempImg1.empty())
//    {   cv::Mat dstimg;
//        int w = size, h = 0;
//        h = tempImg1.rows * size /tempImg1.cols;
//        std::cout<<tempImg1.cols<<tempImg1.rows<<"-"<<dstw<<":"<<dsth<<std::endl;
//        if(h%2)
//            h++;
//        cv::resize(tempImg1, dstimg, cv::Size(w, h), cv::INTER_AREA);
//        tempImg1 = dstimg.clone();
//        dstimg.release();
//    }
//    
//}



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

int VideoDeal::write_file(std::shared_ptr<T_AlarmInfo> info)
{
    int ret = 0;
    int rc  = -1; // 默认失败
    std::string output_filename = info->alarm_file_name;

    AVFormatContext *output_ctx = nullptr;
    AVStream *in_stream  = nullptr;
    AVStream *out_stream = nullptr;

    AVPacket *pkt = nullptr;
    std::shared_ptr<BlockingQueue<AVPacket*>> Rec_Frame_lists2;

    // —— 起点与单调性状态（输入域零点 + 输出域守卫）——
    int64_t base_pts_in  = AV_NOPTS_VALUE;
    int64_t base_dts_in  = AV_NOPTS_VALUE;
    int64_t last_out_pts = AV_NOPTS_VALUE;
    int64_t last_out_dts = AV_NOPTS_VALUE;
    const int64_t OUT_STEP = 1; // 输出时基的最小步长

    int64_t start_pts = AV_NOPTS_VALUE; // 仅用于统计/日志
    int64_t start_dts = AV_NOPTS_VALUE;
    int64_t last_pts  = AV_NOPTS_VALUE;

    bool opened_io = false;
    bool wrote_header = false;
    bool got_keyframe = false;
    bool first_kf_pending = false; // 首个关键帧尚未成功写出（待提交）

    auto free_pkt = [](AVPacket *&p){ if (p) { av_packet_free(&p); p = nullptr; } };

    // —— 安全缩放 + 单调性守卫：写前调用 —— 
    auto validate_fix_ts = [&](AVPacket* p)->bool {
        if (!p || !p->data || p->size <= 0) return false;

        // 归一化到输入域零点
        int64_t in_pts = p->pts, in_dts = p->dts;
        if (base_pts_in != AV_NOPTS_VALUE && in_pts != AV_NOPTS_VALUE) in_pts -= base_pts_in;
        if (base_dts_in != AV_NOPTS_VALUE && in_dts != AV_NOPTS_VALUE) in_dts -= base_dts_in;

        // 缺失互补
        if (in_pts == AV_NOPTS_VALUE && in_dts != AV_NOPTS_VALUE) in_pts = in_dts;
        if (in_dts == AV_NOPTS_VALUE && in_pts != AV_NOPTS_VALUE) in_dts = in_pts;

        // 关系修补：dts ≤ pts
        if (in_pts != AV_NOPTS_VALUE && in_dts != AV_NOPTS_VALUE && in_dts > in_pts) in_dts = in_pts;

        // 缩放到输出时基（带舍入策略）
        int64_t out_pts = AV_NOPTS_VALUE, out_dts = AV_NOPTS_VALUE;
        if (in_pts != AV_NOPTS_VALUE)
            out_pts = av_rescale_q_rnd(in_pts, in_stream->time_base, out_stream->time_base,
                        (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));
        if (in_dts != AV_NOPTS_VALUE)
            out_dts = av_rescale_q_rnd(in_dts, in_stream->time_base, out_stream->time_base,
                        (AVRounding)(AV_ROUND_NEAR_INF | AV_ROUND_PASS_MINMAX));

        // 输出域缺失互补
        if (out_pts == AV_NOPTS_VALUE && out_dts != AV_NOPTS_VALUE) out_pts = out_dts;
        if (out_dts == AV_NOPTS_VALUE && out_pts != AV_NOPTS_VALUE) out_dts = out_pts;

        // 非递减守卫
        if (last_out_dts != AV_NOPTS_VALUE && out_dts != AV_NOPTS_VALUE && out_dts < last_out_dts)
            out_dts = last_out_dts + OUT_STEP;
        if (last_out_pts != AV_NOPTS_VALUE && out_pts != AV_NOPTS_VALUE && out_pts < last_out_pts)
            out_pts = last_out_pts + OUT_STEP;

        // 再保证 dts ≤ pts
        if (out_pts != AV_NOPTS_VALUE && out_dts != AV_NOPTS_VALUE && out_dts > out_pts)
            out_dts = out_pts;

        // duration 兜底
        int64_t out_dur = p->duration > 0 ? av_rescale_q(p->duration, in_stream->time_base, out_stream->time_base) : OUT_STEP;
        if (out_dur <= 0) out_dur = OUT_STEP;

        // 回写
        p->pts = out_pts;
        p->dts = out_dts;
        p->duration = out_dur;
        p->pos = -1;
        p->stream_index = out_stream->index;

        // 维护“上一帧”基准
        if (p->dts != AV_NOPTS_VALUE) last_out_dts = p->dts;
        if (p->pts != AV_NOPTS_VALUE) last_out_pts = p->pts;
        return true;
    };

    // —— 实际写入（失败处理首帧回滚）——
    auto safe_write = [&](AVPacket* p, const char* phase, int frame_i)->int {
        // 记录原始输入域（用于日志）
        int64_t ori_pts = p->pts, ori_dts = p->dts;
        if (base_pts_in  != AV_NOPTS_VALUE && ori_pts != AV_NOPTS_VALUE) ori_pts += base_pts_in;
        if (base_dts_in  != AV_NOPTS_VALUE && ori_dts != AV_NOPTS_VALUE) ori_dts += base_dts_in;

        if (!validate_fix_ts(p)) {
            LOG_WARN(std::string("drop invalid packet before write (empty/pts). file: ") + output_filename);
            return 0; // 当做写入成功跳过
        }

        int wret = av_interleaved_write_frame(output_ctx, p);
        if (wret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE]; av_strerror(wret, errbuf, sizeof(errbuf));
            std::ostringstream oss;
            oss << phase << " writing packet: " << errbuf
                << " file: " << output_filename
                << " ori_pts:" << ori_pts << " ori_dts:" << ori_dts
                << " out_pts:" << p->pts  << " out_dts:" << p->dts
                << " tb_in:"  << in_stream->time_base.num  << "/" << in_stream->time_base.den
                << " tb_out:" << out_stream->time_base.num << "/" << out_stream->time_base.den
                << " / frame_num: " << frame_i;
            LOG_WARN(oss.str());

            if (first_kf_pending) {
                LOG_WARN("first keyframe write failed; reset keyframe state and wait for next keyframe. file: " + output_filename);
                got_keyframe = false;
                first_kf_pending = false;
                base_pts_in = base_dts_in = AV_NOPTS_VALUE;
                last_out_pts = last_out_dts = AV_NOPTS_VALUE;
                output_ctx->start_time = AV_NOPTS_VALUE; // 可选
            }
            return wret;
        }

        if (first_kf_pending) first_kf_pending = false; // 首关键帧成功落盘
        return 0;
    };

    // ========== 2. 创建输出上下文 ==========
    ret = avformat_alloc_output_context2(&output_ctx, NULL, NULL, output_filename.c_str());
    if (ret < 0 || !output_ctx) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE]; av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR(std::string("Failed to allocate output context: ") + errbuf + " file: " + output_filename);
        goto cleanup;
    }

    // ========== 3. 添加视频流 ==========
    in_stream  = pFormatCtx->streams[videoStream_best];
    out_stream = avformat_new_stream(output_ctx, NULL);
    if (!out_stream) {
        LOG_ERROR("Failed to create output stream for file: " + output_filename);
        goto cleanup;
    }

    // ========== 4. 复制编解码参数（含 extradata）==========
    ret = avcodec_parameters_copy(out_stream->codecpar, in_stream->codecpar);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE]; av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR(std::string("Failed to copy codec parameters: ") + errbuf + " file: " + output_filename);
        goto cleanup;
    }
    out_stream->time_base = in_stream->time_base;
    // out_stream->codecpar->codec_tag = 0; // 可选：交给 muxer 决定

    if (out_stream->codecpar->extradata_size <= 0) {
        LOG_WARN("missing codec extradata (SPS/PPS/VPS) on output track, playback may fail. file: " + output_filename);
        // 如需强制稳态，可在解码/拉流侧用 extract_extradata BSF 先行填充
    }

    // ========== 6. 打开输出文件 ==========
    if (!(output_ctx->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&output_ctx->pb, output_filename.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE]; av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_ERROR(std::string("Failed to open output file: ") + errbuf + " file: " + output_filename);
            goto cleanup;
        }
        opened_io = true;
    }

    // 交织缓冲延迟置 0，更稳
    output_ctx->max_interleave_delta = 0;

    // ========== 7. 写文件头 ==========
    ret = avformat_write_header(output_ctx, NULL);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE]; av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR(std::string("Failed to write header: ") + errbuf + " file: " + output_filename);
        goto cleanup;
    }
    wrote_header = true;

    // ========== 从 Rec_Frame_lists 复制快照 ==========
    {
        std::unique_lock<std::mutex> lock(rec_frame_mtx);
        if (Rec_Frame_lists.get() != nullptr) {
            int rec_frame_size = Rec_Frame_lists->GetSize();
            Rec_Frame_lists2.reset(new BlockingQueue<AVPacket*>(rec_frame_size));

            std::vector<AVPacket*> temp_packets;
            temp_packets.reserve(rec_frame_size);

            for (int i = 0; i < rec_frame_size; i++) {
                AVPacket* tmp = nullptr;
                if (Rec_Frame_lists->IsEmpty()) { usleep(1000); continue; }
                if (Rec_Frame_lists->Pop(tmp) == 0) temp_packets.push_back(tmp);
            }
            for (auto &p : temp_packets) {
                AVPacket* copy = av_packet_clone(p);
                if (copy) Rec_Frame_lists2->Push(copy);
                Rec_Frame_lists->Push(p); // 归还原包
            }
        } else {
            LOG_ERROR("Rec_Frame_lists is null, cannot snapshot. file: " + output_filename);
        }
    }
    {
        std::unique_lock<std::mutex> lock(rec_alarm_mtx);
        cur_pkt_update_status[info->algo_name] = false;
        cur_pkt_ids.push_back(info->algo_name);
    }

    // ========== 8. 主循环：处理报警前视频包 ==========
    if (Rec_Frame_lists2 && !Rec_Frame_lists2->IsEmpty()) {
        int rec_size = Rec_Frame_lists2->GetSize();
        for (int i = 0; i < rec_size; i++) {
            if (!Rec_Frame_lists2 || Rec_Frame_lists2->IsEmpty()) break;
            if (Rec_Frame_lists2->Pop(pkt) != 0) break; // pkt 为克隆，需 free
            if (!pkt || !pkt->data || pkt->size <= 0) { free_pkt(pkt); continue; }

            // 只处理指定的视频轨
            if (pkt->stream_index != videoStream_best) { free_pkt(pkt); continue; }

            // 等首个关键帧作为起点（且仅在成功写出后确立）
            if (!got_keyframe) {
                if (!(pkt->flags & AV_PKT_FLAG_KEY)) { free_pkt(pkt); continue; }
                got_keyframe = true;
                first_kf_pending = true;
                base_pts_in = start_pts = pkt->pts;
                base_dts_in = start_dts = pkt->dts;
                last_pts = pkt->pts;
                if (start_pts != AV_NOPTS_VALUE) {
                    output_ctx->start_time = av_rescale_q(start_pts, in_stream->time_base, AV_TIME_BASE_Q);
                }
            }

            ret = safe_write(pkt, "front alarm", i);
            free_pkt(pkt);
            if (ret < 0) continue; // 丢掉坏包/失败，继续后续
        }
    } else if (!Rec_Frame_lists2) {
        LOG_ERROR("Rec_Frame_lists2 not created; no pre-alarm packets. file: " + output_filename);
    }

    // ========== 8b. 主循环：处理报警后视频包 ==========
    for (int i = 0; i < alarFrameCnt; i++) {
        {
            std::unique_lock<std::mutex> lock(rec_alarm_mtx);
            bool ok = rec_alarm_cv.wait_for(
                lock, std::chrono::milliseconds(500),
                [this, &info]{ return cur_pkt_update_status[info->algo_name] || alarm_stop; }
            );
            if (alarm_stop) {
                LOG_ERROR("Stopped by alarm_stop during after-alarm loop. file: " + output_filename);
                break;
            }
            if (!ok) {
                LOG_ERROR("Wait for cur_pkt_update_status timeout for algo: " + info->algo_name + " file: " + output_filename);
                break;
            }
            pkt = av_packet_clone(rec_pkt); // 克隆出的包必须 free
            cur_pkt_update_status[info->algo_name] = false;
        }

        if (!pkt || !pkt->data || pkt->size <= 0) { LOG_WARN("cloned packet empty (after-alarm). file: " + output_filename); free_pkt(pkt); continue; }
        if (pkt->stream_index != videoStream_best) { free_pkt(pkt); continue; }

        if (!got_keyframe) {
            if (!(pkt->flags & AV_PKT_FLAG_KEY)) { free_pkt(pkt); continue; }
            got_keyframe = true;
            first_kf_pending = true;
            base_pts_in = start_pts = pkt->pts;
            base_dts_in = start_dts = pkt->dts;
            last_pts = pkt->pts;
            if (start_pts != AV_NOPTS_VALUE) {
                output_ctx->start_time = av_rescale_q(start_pts, in_stream->time_base, AV_TIME_BASE_Q);
            }
        }

        ret = safe_write(pkt, "after alarm", i);
        free_pkt(pkt);
        if (ret < 0) continue;
    }
	
    // 14. 设置流时长（可选）
    if (start_pts != AV_NOPTS_VALUE && last_pts != AV_NOPTS_VALUE && out_stream) {
        int64_t duration = last_pts - start_pts;
        out_stream->duration = av_rescale_q(duration, in_stream->time_base, out_stream->time_base);
    }

    // 15. 写文件尾 + 刷出
    if (wrote_header) {
        int tr = av_write_trailer(output_ctx);
        if (tr < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(tr, errbuf, sizeof(errbuf));
            LOG_ERROR(std::string("av_write_trailer failed: ") + errbuf + " file: " + output_filename);
            rc = (rc == 0 ? tr : rc);
        }
        if (output_ctx && output_ctx->pb) avio_flush(output_ctx->pb);
        rc = (rc == -1) ? 0 : rc; // 若前面无错误，把 rc 置 0
    }

cleanup:

    // 清空 Rec_Frame_lists2 中未写的克隆包
    if (Rec_Frame_lists2) {
        while (!Rec_Frame_lists2->IsEmpty()) {
            AVPacket* p = nullptr;
            if (Rec_Frame_lists2->Pop(p) == 0) { if (p) av_packet_free(&p); }
        }
        Rec_Frame_lists2.reset();
    }

    // 释放当前循环中的 pkt（若还没释放）
    free_pkt(pkt);

    if (opened_io && output_ctx && !(output_ctx->oformat->flags & AVFMT_NOFILE)) {
        int clo = avio_closep(&output_ctx->pb);
        if (clo < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(clo, errbuf, sizeof(errbuf));
            LOG_ERROR(std::string("avio_closep failed: ") + errbuf + " file: " + output_filename);
            rc = (rc == 0 ? clo : rc);
        }
        opened_io = false;
    }

    if (output_ctx) { avformat_free_context(output_ctx); output_ctx = nullptr; }

    {
        std::unique_lock<std::mutex> lock(rec_alarm_mtx);
        cur_pkt_update_status.erase(info->algo_name);
        auto it = std::find(cur_pkt_ids.begin(), cur_pkt_ids.end(), info->algo_name);
        if (it != cur_pkt_ids.end()) cur_pkt_ids.erase(it);
    }

    info->alarm_status = HANDLERED;

    if (rc != 0) {
        LOG_ERROR("Write file failed (rc=" + std::to_string(rc) + "): " + output_filename);
    }else{
        LOG_DEBUG("Write file succese (rc=" + std::to_string(rc) + "): " + output_filename);
	}
    return rc;
}


void VideoDeal::alarmVideo()
{
	alarm_stop = false;
	while (b_getter && end_of_stream){
		
//		std::cout << "[INFO] === video hander alarm msg by algo_name " << video_path << std::endl;
		
		bool b_alarm_info_update = AlarmVideoInfo::WaitAlarmInfoUpdate(video_path);

		if (!b_alarm_info_update) continue;

		{
			std::unique_lock<std::mutex> lock(AlarmVideoInfo::alarm_mutex);
			for (auto &info : AlarmVideoInfo::alarm_infos){
				if (info->rtsp_path == video_path && info->alarm_status == NEED_HANDLER){
					// cur_pkt_update_status待优化，需要及时释放

					info->alarm_status = HANDLERING;
					std::thread t_alarm_video = std::thread(&VideoDeal::write_file, this, info);
					t_alarm_video.detach();
					
					std::cout << "[INFO] === video hander alarm msg by algo_name : " << info->algo_name << std::endl;
				}
			}

		}

	}

}


void VideoDeal::getPacket()
{
    #ifdef DebugOpen
    std::cout<<"get packet start!"<<std::endl;
    #endif
    AVPacket *packet = av_packet_alloc(); //分配一个packet
    avcodec_flush_buffers(pCodecCtx);
    end_of_stream = 1;
    bool firstIpacket = true;
    int IpacketCnt = 0;

    auto start = std::chrono::system_clock::now();
    auto begin = std::chrono::system_clock::now();

    while(b_getter && end_of_stream)
    {
        // std::cout<<"in getPacket thread!"<<std::endl;
        if (isFile)
        {
            // usleep(0.8/av_q2d(pFormatCtx->streams[videoStream_best]->r_frame_rate)*1000000);
            // usleep(1000000/fps*0.6);
            usleep(1000000/fps);
            // usleep(1000000/fps*2); //fish_tdds
            // usleep(25*1000);
            // usleep(100*1000);
            // usleep(100*1000);
        }
        time(&TimeNow);
        int ret_read = av_read_frame(pFormatCtx, packet);
        if (ret_read != 0)
        {
            if(0 && -541478725 == ret_read && countOfEof < 5)
            {
                // countOfEof++;
                std::cout<<"[Warning] read EOF, may it is not end of stream! "<<video_path<<std::endl;
                usleep(100000);
                continue;
            }
            else
            {
                std::cout<<"end_of_stream,code"<<ret_read<<",reason:"<<std::endl;
                char str[200] = {0};
                av_strerror(ret_read,str,200);
                printf("%s",str);
                av_packet_unref(packet);
                end_of_stream = 0;
                // if(isFile)
                // {
                //     while(packet_list.get()!=nullptr && packet_list->GetSize()>0)
                //     {
                //         usleep(100000);
                //         std::cout<<"File end_of_stream!"<<std::endl;
                //     }
                // }
                // break;
                continue;
            }
        }
        else
        {
            countOfEof = 0;
        }
        if(isPause)
        {
            av_packet_unref(packet);
            usleep(10000);
            continue;
        }
      
        if(packet->size <= 0 || packet->stream_index != videoStream_best)
        {
            av_packet_unref(packet);
            continue;
        }

		// 为每个算法提供实时视频帧
        {
	        std::unique_lock<std::mutex> lock(rec_alarm_mtx);
			if (!cur_pkt_ids.empty() && !alarm_stop){
				
				if (rec_pkt) av_packet_free(&rec_pkt);
				
				rec_pkt = av_packet_clone(packet);
				if (rec_pkt) {
					
					for(auto &pkt_id : cur_pkt_ids){
						cur_pkt_update_status[pkt_id] = true;
					}
					rec_alarm_cv.notify_all();
				}
			}
		}

        //录像队列
        {
        	std::unique_lock<std::mutex> lock(rec_frame_mtx);
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
                std::cout<<"packetList size:"<<packetSize<<",dump size:"<<dumpPackCnt<<","<<video_path<<std::endl;
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
        // begin = std::chrono::system_clock::now();
        // std::cout<<"[INFO] get packet use:"<<(int)(std::chrono::duration_cast<std::chrono::milliseconds>(begin - start).count())<<"\n";
        // start = std::chrono::system_clock::now();
    }
    av_packet_unref(packet);
    av_packet_free(&packet);
    #ifdef DebugOpen
    std::cout<<"get packet out!"<<std::endl;
    #endif
}

void VideoDeal::getHwFrame()
{
    AVFrame *hwFrame = NULL;
    AVFrame *sw_Frame = av_frame_alloc();
    
    while(b_getter && end_of_stream)
    {
        if(hwFrame_list.get()!= nullptr && hwFrame_list->GetSize()>=1)
        {
            if(hwFrame_list->Pop(hwFrame,100) != 0)
            {
                // std::cout<<"hwFrame_list pop fail"<<std::endl;
                usleep(5000);
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
                        printf("[WARN] drop this error,otherwise will cause av_hwframe_transfer_data() crash ,%s\n",video_path.c_str());
                        continue;
                    }
                    else
                    {
                        testH = hwFrame->height;
                        testW = hwFrame->width;
                    }
                }
            }

            nRet = av_hwframe_transfer_data(sw_Frame, hwFrame, 0);
            // nRet = av_hwframe_map(sw_Frame, hwFrame, 0);
            if (nRet != 0) {
                printf("av_hwframe_transfer_data fail %d",nRet);
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
                av_free(tmp);
            }
            av_frame_unref(hwFrame);
            av_frame_free(&hwFrame);
            av_frame_unref(sw_Frame);
            usleep(5000);
        }
        else
        {
            usleep(5000);
        }
        
    }
    avcodec_flush_buffers(pCodecCtx);
    av_frame_unref(sw_Frame);
    av_frame_free(&sw_Frame);
    av_frame_unref(hwFrame);
    av_frame_free(&hwFrame);
    av_free(hwFrame);
    av_free(sw_Frame);
    #ifdef DebugOpen
    std::cout<<"getHwFrame th out"<<std::endl;
    #endif
}

void VideoDeal::resize_frame()
{
    while(b_getter && end_of_stream)
    {
        cv::Mat img;
        if(resize_in_list->GetSize() <= 0)
        {
            usleep(5000);
            continue;
        }
        if(resize_in_list->Pop(img,100) == 0)
        {
            // struct timeval tv1,tv2;
            // gettimeofday(&tv1,NULL);
            if(isReSize)
                sw_resize(img);
            // gettimeofday(&tv2,NULL);
            // std::cout<<"resize: "<<(tv2.tv_sec - tv1.tv_sec)*1000 + (tv2.tv_usec - tv1.tv_usec)/1000<<"\n";
            resize_out_list->Push(img);
        }
        
    }
    #ifdef DebugOpen
    std::cout<<"resize_frame out"<<"\n";
    #endif
}

void VideoDeal::getter()
{
    int index = 1;
    int ret = 0;
    AVFrame *pFrameRGB = av_frame_alloc();
    AVFrame *sw_Frame = av_frame_alloc();
    //=========================== 分配AVPacket结构体 ===============================//
    // AVPacket *packet = av_packet_alloc(); //分配一个packet
    // avcodec_flush_buffers(pCodecCtx);
    //===========================  读取视频信息 ===============================//
    int got_frame = 0;

    end_of_stream = 1;

    std::thread t_getPacketer =  std::thread(&VideoDeal::getPacket,this);

    std::thread t_getHwFrame =  std::thread(&VideoDeal::getHwFrame,this);

	std::thread t_alarmVideo = std::thread(&VideoDeal::alarmVideo, this);

    std::thread resize_th(&VideoDeal::resize_frame,this);
    
    struct timeval tv1,tv2,tv3,tv4;
    int nRet = 0; 
    while (b_getter && end_of_stream)
    {       
        AVPacket *packet = 0;
        if (packet_list->GetSize() > 0 && packet_list->Pop(packet,100) == 0)
        {
            got_frame = 0;    
            nRet = avcodec_send_packet(pCodecCtx, packet);
            if (nRet != 0)
            {
                char str[200] = {0};
                av_strerror(nRet,str,200);
                // printf("%s",str);
            }
            av_packet_unref(packet);
            av_packet_free(&packet);
            packet = NULL;
            nRet = avcodec_receive_frame(pCodecCtx, pFrameRGB);
            if (nRet != 0)
            {
                av_frame_unref(pFrameRGB);
                char str[200] = {0};
                av_strerror(nRet,str,200);
                // printf("%s\n",str);
            }

            if (nRet == 0)
            {
                got_frame = 1;

                if(hw_decode)
                {
                    AVFrame* tmp = av_frame_clone(pFrameRGB);
                    av_frame_unref(pFrameRGB);
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
                    AVFrame* tmp = av_frame_clone(pFrameRGB);
                    av_frame_unref(pFrameRGB);
                    if(frame_list.get() != nullptr && frame_list->Push(tmp) != 0)
                    {
                        av_frame_unref(tmp);
                        av_frame_free(&tmp);
                        av_free(tmp);
                    }
                }
                
                
                if(packet)
                {
                    av_packet_unref(packet);
                    av_packet_free(&packet);
                    packet = NULL;
                }
                av_frame_unref(pFrameRGB);
                av_frame_unref(sw_Frame);
                continue;
            }

        }
        
        if(packet)
        {
            av_packet_unref(packet);
            av_packet_free(&packet);
            packet = NULL;
        }
        av_frame_unref(pFrameRGB);
        av_frame_unref(sw_Frame);
        usleep(5000);
    } 
    nRet = 0;
    // while((nRet = avcodec_receive_frame(pCodecCtx, pFrameRGB)) == 0)
    // {
    //     std::cout<<"[WARNING] dump decoder left frame!"<<std::endl;
    //     av_frame_unref(pFrameRGB);
    // }

    #ifdef DebugOpen
    std::cout<<"decode th out begin!"<<std::endl;
    #endif
    b_getter = false;
	
    {
		std::unique_lock<std::mutex> lock(rec_alarm_mtx);
		alarm_stop = true;
		rec_alarm_cv.notify_all();
	}
	AlarmVideoInfo::StopAlarmInfo(video_path);
	
    av_frame_unref(sw_Frame);
    av_frame_unref(pFrameRGB);
    av_frame_free(&pFrameRGB);
    av_frame_free(&sw_Frame);
    av_free(pFrameRGB);
    av_free(sw_Frame);

    
    if (t_getPacketer.joinable())
    {
        t_getPacketer.join();
    }

    if (t_getHwFrame.joinable())
    {
        t_getHwFrame.join();
    }

	if (t_alarmVideo.joinable())
	{
		t_alarmVideo.join();
	}

	if(resize_th.joinable())
    {
        resize_th.join();
    }

    resize_in_list->Clear();
    resize_out_list->Clear();
    resize_in_list.reset();
    resize_out_list.reset();
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
    }
    packet_list.reset();
    if(hwFrame_list.get()!= nullptr)
    {
        avcodec_flush_buffers(pCodecCtx);
        AVFrame* tmp = NULL;
        while(hwFrame_list->GetSize() > 0)
        {
            hwFrame_list->Pop(tmp,100);
            av_frame_unref(tmp);
            av_frame_free(&tmp);
            av_free(tmp);

            tmp = NULL;
        }
        hwFrame_list->Clear();
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
            av_free(tmp);
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
            av_free(tmp);
            tmp = NULL;
        }
        frame_list->Clear();
    }
    frame_list.reset();

    avcodec_flush_buffers(pCodecCtx);
    avcodec_close(pCodecCtx); //关闭视频解码器
    avcodec_free_context(&pCodecCtx);

    avformat_flush(pFormatCtx);
    try
    {
        avformat_close_input(&pFormatCtx);
        //avformat_free_context(pFormatCtx);
    } //关闭输入文件
    catch (...)
    {
        std::cout<<"input close error"<<std::endl;
    }

    pCodecCtx = nullptr;

    #ifdef DebugOpen
    std::cout<<"decode thread is down!:"<<b_getter<<status<<end_of_stream<<std::endl;
    #endif
}

static int InterruptFouction(void *theTimeSpec)
{
    if (!theTimeSpec) {
        return 0;
    }
    
    // std::cout << "InterruptFouction called~!" << std::endl;
    time_t mtime;
    time(&mtime);
    time_t *t = (time_t*)theTimeSpec;
    
    if( ( mtime - *t ) > 10) {
        printf("rtsp time out restart !\n");
        return 1;

    }
    return 0;
}


int VideoDeal::main_ffmpeg(const char *file_path)
{
    time(&TimeNow);
    g_file_path = file_path;

    pFormatCtx = avformat_alloc_context();

    pFormatCtx->interrupt_callback.callback = InterruptFouction;
    pFormatCtx->interrupt_callback.opaque = &TimeNow;

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

    AVDictionary *avDictionary = NULL;
    // if(!isFile)
    //     av_dict_set(&avDictionary, "fflags", "nobuffer", 0);   // 防压手可以不用设置av_dict_set，会导致延迟
    av_dict_set(&avDictionary, "probesize", "5760000", 0); // 2048 初始化没影响延迟
    av_dict_set(&avDictionary, "max_analyze_duration", "10", 0);
    // av_dict_set(&avDictionary, "listen_timeout", "10000000", 0);

    av_dict_set(&avDictionary, "stimeout", "10000000", 0); //10s
    av_dict_set(&avDictionary, "rw_timeout", "10000000", 0); //ms          
    av_dict_set(&avDictionary, "buffer_size", "5760000", 0);
    av_dict_set(&avDictionary, "rtsp_transport", "tcp", 0); // 设置使用tcp传输,因udp在1080p下会丢包导致花屏
    int ret = avformat_open_input(&pFormatCtx, file_path, nullptr, &avDictionary);
    av_dict_free(&avDictionary);

    if (ret != 0)
    {
        char str[200] = {0};
        av_strerror(ret,str,200);
        printf("av_strerror:%s \n",str);
        std::string s(str);
        errMsg=s;
        avformat_close_input(&pFormatCtx);   
        return -1;
    }
    //=================================== 获取视频流信息 ===================================//
    if (avformat_find_stream_info(pFormatCtx, NULL) < 0)
    {
        avformat_close_input(&pFormatCtx);
        printf("Could't find stream infomation.");
        return -1;
    }
    videoStream_best = av_find_best_stream(pFormatCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (videoStream_best < 0)
    {
        avformat_close_input(&pFormatCtx);
        printf("Didn't find a video stream.");
        return -1;
    }
    //=================================  查找解码器 ===================================//
    width = pFormatCtx->streams[videoStream_best]->codecpar->width;
    height = pFormatCtx->streams[videoStream_best]->codecpar->height;
    fps = av_q2d(pFormatCtx->streams[videoStream_best]->r_frame_rate);
    if(fps <= 0 || fps > 50)
        fps = 25;
    alarFrameCnt = fps * 6;//报警录像时长

    // frame_list.reset(new BlockingQueue<AVFrame*>(2));
    // Rec_Frame_lists.reset(new BlockingQueue<AVPacket*>((int)(alarFrameCnt)));
    // hwFrame_list.reset(new BlockingQueue<AVFrame*>(3));
    // resize_in_list.reset(new BlockingQueue<cv::Mat>(2));
    // resize_out_list.reset(new BlockingQueue<cv::Mat>(3));
    // packet_list.reset(new BlockingQueue<AVPacket*>(fps * 3));

    // frame_list.reset(new BlockingQueue<AVFrame*>(8));
    // Rec_Frame_lists.reset(new BlockingQueue<AVPacket*>((int)(alarFrameCnt)));
    // hwFrame_list.reset(new BlockingQueue<AVFrame*>(8));
    // resize_in_list.reset(new BlockingQueue<cv::Mat>(8));
    // resize_out_list.reset(new BlockingQueue<cv::Mat>(8));
    // packet_list.reset(new BlockingQueue<AVPacket*>(fps * 8));

    // frame_list.reset(new BlockingQueue<AVFrame*>(2));
    // Rec_Frame_lists.reset(new BlockingQueue<AVPacket*>((int)(alarFrameCnt)));
    // hwFrame_list.reset(new BlockingQueue<AVFrame*>(2));
    // resize_in_list.reset(new BlockingQueue<cv::Mat>(2));
    // resize_out_list.reset(new BlockingQueue<cv::Mat>(2));
    // packet_list.reset(new BlockingQueue<AVPacket*>(2)); //不要延迟，很多需要实时响应，给防压手

    frame_list.reset(new BlockingQueue<AVFrame*>(2));
    Rec_Frame_lists.reset(new BlockingQueue<AVPacket*>((int)(alarFrameCnt)));
    hwFrame_list.reset(new BlockingQueue<AVFrame*>(3));
    resize_in_list.reset(new BlockingQueue<cv::Mat>(2));
    resize_out_list.reset(new BlockingQueue<cv::Mat>(3));
    packet_list.reset(new BlockingQueue<AVPacket*>(5));
    // packet_list.reset(new BlockingQueue<AVPacket*>(fps * 3));


    // #ifdef DebugOpen
    std::cout<<"[INFO] start rtsp fps:"<<fps<<" alarFrameCnt:"<<alarFrameCnt<<file_path_str<<std::endl;
    // #endif
    pCodecCtx = avcodec_alloc_context3(NULL);
    avcodec_parameters_to_context(pCodecCtx, pFormatCtx->streams[videoStream_best]->codecpar);
    pCodecCtx -> thread_count = 2;
    const AVCodec *pCodec;
    
    if(hw_decode)
    {
        if(pCodecCtx->codec_id == AV_CODEC_ID_H265)
            pCodec = avcodec_find_decoder_by_name("hevc_rkmpp");
        else if(pCodecCtx->codec_id == AV_CODEC_ID_H264)
            pCodec = avcodec_find_decoder_by_name("h264_rkmpp");
        else if(pCodecCtx->codec_id == AV_CODEC_ID_H263)
            pCodec = avcodec_find_decoder_by_name("h263_rkmpp");
        else if(pCodecCtx->codec_id == AV_CODEC_ID_MPEG1VIDEO)
            pCodec = avcodec_find_decoder_by_name("mpeg1_rkmpp");
        else if(pCodecCtx->codec_id == AV_CODEC_ID_MPEG1VIDEO)
            pCodec = avcodec_find_decoder_by_name("mpeg2_rkmpp");
        else if(pCodecCtx->codec_id == AV_CODEC_ID_MPEG4)
            pCodec = avcodec_find_decoder_by_name("mpeg4_rkmpp");
        else if(pCodecCtx->codec_id == AV_CODEC_ID_VP8)
            pCodec = avcodec_find_decoder_by_name("vp8_rkmpp");
        else if(pCodecCtx->codec_id == AV_CODEC_ID_VP9)
            pCodec = avcodec_find_decoder_by_name("vp9_rkmpp");
        else
            pCodec= avcodec_find_decoder(pCodecCtx->codec_id);
    }
    else
    {
        if(pCodecCtx->codec_id == AV_CODEC_ID_H265)//AV_CODEC_ID_H264
            pCodec = avcodec_find_decoder_by_name("hevc");
        else if(pCodecCtx->codec_id == AV_CODEC_ID_H264)
            pCodec = avcodec_find_decoder_by_name("h264");
        else
            pCodec= avcodec_find_decoder(pCodecCtx->codec_id);
    }
    
    if (&pCodec == NULL)
    {
        printf("Codec not found.");
        return -1;
    }
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



VideoDeal::VideoDeal(std::string videoPath, int gpu_device):
video_path(videoPath),
t_getter(nullptr),
b_getter(false),
end_of_stream(0),
countOfEof(0),
isPause(false)
{

    #ifdef DECNOTRESIZE
    isReSize = false;
    #else
    isReSize = true;
    #endif

    av_log_set_level(AV_LOG_QUIET);
    // av_log_set_level(AV_LOG_DEBUG);
    // av_register_all();    
    avformat_network_init(); // 网络初始化
    
    b_getter = false;
    #ifdef DebugOpen
    printf("video path : %s\n", videoPath.c_str());
    #endif
    int ret = 1;
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
        printf("\nvideo open error\n");
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
    std::cout<<"~VideoDeal() begin:"<<video_path<<std::endl;  
    std::lock_guard<std::mutex> lock(release_vid); //推流的rtsp断了会卡住，有没有这个锁都一样
    b_getter = false;
	alarm_stop = true;

    if (t_getter != nullptr && t_getter->joinable())
    {
        t_getter->join();
        delete t_getter;
    } 
    std::cout<<"~VideoDeal() end:"<<video_path<<std::endl;    
}


cv::Mat ConvertMat(AVFrame* pFrame, int width, int height)
{   //nv12（yuv420sp）存储格式为yyyuvuvuv,在AVFrame中data[0]存yyy,data[1]存uvuvuv
    //(yuv420p)则为yyy uuu vvv 分别对应data[0],[1],[2]
    //rgb32则为data[0]
    struct timeval tv1,tv2,tv3,tv4;
    // gettimeofday(&tv3,NULL);
    cv::Mat bgr= cv::Mat::zeros( height, width, CV_8UC3);
    cv::Mat tmp_img = cv::Mat::zeros( height*3/2, width, CV_8UC1); 
    // printf("\nbegin ConvertMat,tmp_img size:%d\n",tmp_img.size());   
    memcpy( tmp_img.data, pFrame->data[0], width*height);
    memcpy( tmp_img.data + width*height, pFrame->data[1], width*height/2);
    //memcpy( tmp_img.data + width*height*5/4, pFrame->data[2], width*height/4 );
    cv::cvtColor(tmp_img, bgr, cv::COLOR_YUV2BGR_NV12);

    av_frame_unref(pFrame);
    // av_frame_free(&pFrame);
    tmp_img.release();
    // gettimeofday(&tv4,NULL);
    // std::cout<<"get frame use:"<<(tv4.tv_sec - tv3.tv_sec)*1000 + (tv4.tv_usec - tv3.tv_usec)/1000<<"->"<<std::endl;
    return bgr;
}

int VideoDeal::getframe(cv::Mat &ansMat)
{
    struct timeval tv1,tv2,tv3,tv4;
    if(b_getter && end_of_stream)
    {
        // std::cout<<"packet_list:"<<packet_list->GetSize()<<"hwFrame_list:"<<hwFrame_list->GetSize()<<"frame_list:"<<frame_list->GetSize()<<"--resize_in_list:"<<resize_in_list->GetSize()<<"--resize_out_list:"<<resize_out_list->GetSize()<<"\n"; 
        // packet_list:0hwFrame_list:0frame_list:8--resize_in_list:0--resize_out_list:2

        AVFrame* frame = NULL;
        if(frame_list.get()!= nullptr && frame_list->GetSize() == 0)
            usleep(1000);
        
        if(frame_list.get()!= nullptr && frame_list->GetSize()>=1)
        {
            cv::Mat mat;
            if(frame_list->Pop(frame,100) != 0)
            {
                av_frame_unref(frame);
                av_frame_free(&frame);
                usleep(1000);
                return 1;
            }
            gettimeofday(&tv1,NULL);
            mat = ConvertMat(frame,frame->width,frame->height);
            gettimeofday(&tv2,NULL);
            // std::cout<<"toMat:"<<(tv2.tv_sec - tv1.tv_sec)*1000 + (tv2.tv_usec - tv1.tv_usec)/1000<<"\n"; //才2ms
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
                    // if(resize_out_list->IsFull())
                    //     resize_out_list->Pop(ansMat,100);
                    if(resize_out_list.get()!= nullptr && resize_out_list->Pop(ansMat,100) == 0)
                        return 0;
                    else
                    {
                        usleep(1000);
                        return 1;
                    }
                }
            }
        }
        //std::cout<<"---------------------list is empty! ---------------------------"<<std::endl;
        return 1; //list is empty
    }
    std::cout<<"---------------------decode thread is dead! ----------------------:"<<b_getter<<","<<end_of_stream<<std::endl;
    return -2; //thread is dead
}

AVPacket* VideoDeal::getPacketFromQue()
{
    AVPacket *packet = NULL;
    if(Rec_Frame_lists.get() != nullptr && Rec_Frame_lists->GetSize()>0)
        Rec_Frame_lists->Pop(packet,100);
    return packet;
}


