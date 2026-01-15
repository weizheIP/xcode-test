
#pragma once

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
extern "C"
{
#include <stdio.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>
}

static int InterruptFouction(void *theTimeSpec)
{
    if (!theTimeSpec) {
        return 0;
    }
    time_t mtime;
    time(&mtime);
    time_t *t = (time_t*)theTimeSpec;    
    if( ( mtime - *t ) > 10) {
        printf("[ERROR] rtsp time out restart !\n");
        return 1;
    }
    return 0;
}

class VideoDeal
{
public:
    VideoDeal(const std::string &input_file, int gpu_id = 0) : inputFile(input_file)
    {
        status = initialize();
    }
    
    ~VideoDeal()
    {
        cleanup();
        std::cout << "[INFO] cap release: " << inputFile << std::endl;
    }

    void cleanup() {        
        if (decoderCtx) {
            avcodec_flush_buffers(decoderCtx); //qk
            avcodec_free_context(&decoderCtx);
            decoderCtx = nullptr;
        }
        if (inputCtx) {
            avformat_flush(inputCtx);
            avformat_close_input(&inputCtx);
            inputCtx = nullptr;
        }
        avformat_network_deinit(); // 清理网络相关资源
    }

    bool initialize()
    {
        avformat_network_init();

        inputCtx = avformat_alloc_context();
        time(&TimeNow);
        inputCtx->interrupt_callback.callback = InterruptFouction;
        inputCtx->interrupt_callback.opaque = &TimeNow;

        AVDictionary *options = nullptr;
        av_dict_set(&options, "buffer_size", "2048000", 0); //2048的话3k花屏
        av_dict_set(&options, "rtsp_transport", "tcp", 0); //不知道会不会导致卡住，udp的时候没卡过，可以加interrupt_callback防止阻塞或者rw_timeout
        // av_dict_set(&options, "rtsp_transport", "udp", 0); 
        av_dict_set(&options, "stimeout", "10000000", 0); //10s rtsp
        // av_dict_set(&options, "listen_timeout", "10", 0);
        av_dict_set(&options, "timeout", "10000000", 0); //10s udp或者http
        av_dict_set(&options, "rw_timeout", "10000", 0); //ms        
        av_dict_set(&options, "max_delay", "500000", 0);
        // av_dict_set(&options, "probesize", "5760000", 0); // 2048
        // av_dict_set(&options, "max_analyze_duration", "10", 0);
        // av_dict_set(&options, "fflags", "nobuffer", 0); //对阻塞没有用

        int ret = avformat_open_input(&inputCtx, inputFile.c_str(), nullptr, &options);
        av_dict_free(&options);  // 立即释放字典        

        if (ret != 0) {
            std::cerr << "Cannot open input file '" << inputFile << "'" << std::endl;
            cleanup();
            return false;
        }

        if (avformat_find_stream_info(inputCtx, nullptr) < 0) {
            std::cerr << "Cannot find input stream information." << std::endl;
            cleanup();
            return false;
        }

        videoStream = av_find_best_stream(inputCtx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if (videoStream < 0) {
            std::cerr << "Cannot find a video stream in the input file" << std::endl;
            cleanup();
            return false;
        }

        AVCodecParameters *codec_params = inputCtx->streams[videoStream]->codecpar;
        const AVCodec *decoder = NULL;
        
        if (codec_params->codec_id == AV_CODEC_ID_H264) {
            decoder = avcodec_find_decoder_by_name("h264_rkmpp");
        } else if (codec_params->codec_id == AV_CODEC_ID_HEVC) {
            decoder = avcodec_find_decoder_by_name("hevc_rkmpp");
        } else {
            fprintf(stderr, "不支持的编解码器类型\n");
            cleanup();
            return false;
        }
        
        if (!decoder) {
            fprintf(stderr, "找不到合适的解码器\n");
            cleanup();
            return false;
        }

        decoderCtx = avcodec_alloc_context3(decoder);
        if (!decoderCtx) {
            std::cerr << "Failed to allocate codec context" << std::endl;
            cleanup();
            return false;
        }

        if (avcodec_parameters_to_context(decoderCtx, codec_params) < 0) {
            std::cerr << "Failed to copy codec parameters to context" << std::endl;
            cleanup();
            return false;
        }

        if (avcodec_open2(decoderCtx, decoder, nullptr) < 0) {
            std::cerr << "Failed to open codec for stream #" << videoStream << std::endl;
            cleanup();
            return false;
        }
        avcodec_flush_buffers(decoderCtx); //qk
        return true;
    }

    int decodeWrite(AVPacket *packet, cv::Mat &frame)
    {
        AVFrame *hwFrame = nullptr;
        AVFrame *swFrame = nullptr;
        uint8_t *buffer = nullptr;
        cv::Mat yuvImage;
        int ret = 0;
        int size = 0;

        ret = avcodec_send_packet(decoderCtx, packet);
        if (ret < 0) {
            std::cerr << inputFile <<" Error during decoding" << std::endl;
            return ret;
        }

        while (1) {
            hwFrame = av_frame_alloc();
            swFrame = av_frame_alloc();
            
            if (!hwFrame || !swFrame) {
                std::cerr << "Can not alloc frame" << std::endl;
                ret = AVERROR(ENOMEM);
                goto fail;
            }

            ret = avcodec_receive_frame(decoderCtx, hwFrame);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) { //AVERROR_EOF文件结束，EAGAIN还没解码完再试一下
                ret = 0;  // 不是错误情况
                goto fail;
            } else if (ret < 0) {
                std::cerr << inputFile << " Error while decoding" << std::endl;
                goto fail;
            }
            // // chroma_format_idc=59 6 32 默认是1(4:2:0)
            // if (frame->format == AV_PIX_FMT_NONE || frame->format != 1) {
            //     // Log and skip frames with unsupported chroma format
            //     std::cerr << "Unsupported chroma format detected, skipping frame..." << std::endl;
            //     goto fail;
            // }

            if ((ret = av_hwframe_transfer_data(swFrame, hwFrame, 0)) < 0) {
                std::cerr << "Error transferring the data to system memory" << std::endl;
                goto fail;
            }

            size = av_image_get_buffer_size(AVPixelFormat(swFrame->format), 
                                              swFrame->width, 
                                              swFrame->height, 1);
            buffer = (uint8_t *)av_malloc(size);
            if (!buffer) {
                std::cerr << "Can not alloc buffer" << std::endl;
                ret = AVERROR(ENOMEM);
                goto fail;
            }

            ret = av_image_copy_to_buffer(buffer, size,
                                        (const uint8_t *const *)swFrame->data,
                                        (const int *)swFrame->linesize,
                                        AVPixelFormat(swFrame->format),
                                        swFrame->width, swFrame->height, 1);

            if (ret < 0) {
                std::cerr << "Can not copy image to buffer" << std::endl;
                goto fail;
            }

            yuvImage = cv::Mat((swFrame->height) * 3 / 2, swFrame->width, CV_8UC1, buffer);
            cv::cvtColor(yuvImage, frame, cv::COLOR_YUV2BGR_NV12);

            // 成功处理完图像后，释放资源并返回
            av_frame_free(&hwFrame);
            av_frame_free(&swFrame);
            av_freep(&buffer);
            return 0;

        fail:
            if (hwFrame) av_frame_free(&hwFrame);
            if (swFrame) av_frame_free(&swFrame);
            if (buffer) av_freep(&buffer);
            return ret;
        }
    }

    bool getframe(cv::Mat &mat)
    {
        AVPacket packet;
        cv::Mat frame;
        int ret;

        while (1) {
            time(&TimeNow);
            ret = av_read_frame(inputCtx, &packet);
            if (ret < 0) {
                break;
            }

            if (videoStream == packet.stream_index) {
                ret = decodeWrite(&packet, frame);
                av_packet_unref(&packet);  // 无论成功失败都要释放packet
                
                if (ret < 0) {
                    std::cerr << "Error decoding frame" << std::endl;
                    return false;
                }
                
                if (!frame.empty()) {
#ifdef DECNOTRESIZE
                    mat = frame.clone();
#else
                    int w = 1280, h = 0;
                    h = frame.rows * 1280 / frame.cols;
                    if (h % 2) h++;
                    cv::resize(frame, mat, cv::Size(w, h));
#endif
                    return true;
                }
            } else {
                av_packet_unref(&packet);
            }
        }

        return false;
    }

    bool isOpened() const { return inputCtx != nullptr && decoderCtx != nullptr; }
    int alarFrameCnt = 0;
    int fps;
    bool status = false;
    std::string errMsg = "";

private:
    std::string inputFile;
    AVFormatContext *inputCtx = nullptr;
    AVCodecContext *decoderCtx = nullptr;
    int videoStream = -1;
    time_t TimeNow;
};
