

#ifndef _MP4_REC
#define _MP4_REC

// #include <opencv2/opencv.hpp>
#include <iostream>
#include <mutex>
#include <unistd.h>
#include <string>

#include "minimp4.h"
/*
    该类用来把h264/265 nalu 封装成mp4,放弃使用ffmpeg封装,该类单线程
*/


class Mp4Rec
{
public:
Mp4Rec(int w,int h,int fps);
~Mp4Rec();


int start_rec(std::string fileName);
int stop_rec();
int write_one_frame(uint8_t *buf_h264,uint64_t size);


private:
bool _start;
int _fps;
bool _isHevc;
int _height;
int _width;
int _sequentialMode;
int _fragmentationMode;

MP4E_mux_t *_mux;
mp4_h26x_writer_t _mp4wr;
FILE *_fout;
};

#endif