
#define MINIMP4_IMPLEMENTATION

#include "mp4Rec.h"

using namespace std;

static int write_callback(int64_t offset, const void *buffer, size_t size, void *token)
{
    FILE *f = (FILE*)token;
    fseek(f, offset, SEEK_SET);
    return fwrite(buffer, 1, size, f) != size;
}

Mp4Rec::Mp4Rec(int w,int h,int fps)
:_width(w),
_height(h),
_fps(fps),
_start(false),
_isHevc(false),
_mux(nullptr),
_sequentialMode(0),
_fragmentationMode(0),
_fout(NULL)
{
    
}

Mp4Rec::~Mp4Rec()
{
    if(_mux != nullptr)
    {
        MP4E_close(_mux);
        mp4_h26x_write_close(&_mp4wr);
    }
    if (_fout)
    {
        fclose(_fout);
        _fout = NULL;
    }
    _start = false;
}

int Mp4Rec::start_rec(string fileName)
{
    if(_start)
    {
        printf("[ERROR] rec alreadly start!\n");
        stop_rec();
        _start = false;
    }
    _fout = fopen(fileName.c_str(), "wb");
    if (!_fout)
    {
        printf("[ERROR] can't open rec output file:%s!\n",fileName.c_str());
        return -1;
    }
    int is_hevc = _isHevc;

    _mux = MP4E_open(_sequentialMode, _fragmentationMode, _fout, write_callback);

    if (MP4E_STATUS_OK != mp4_h26x_write_init(&_mp4wr, _mux, _width, _height, is_hevc))
    {
        printf("error: mp4_h26x_write_init failed\n");
        return -2;
    }

    printf("[INFO] rec start !!!!!!!!!!!!!\n");
    _start = true;
    return 0;
}

int Mp4Rec::stop_rec()
{
    if(_mux)
    {
        MP4E_close(_mux);
        _mux = nullptr;
        mp4_h26x_write_close(&_mp4wr);
    }
    if (_fout)
    {
        fclose(_fout);
        _fout = NULL;
    }
    _start = false;
    printf("[INFO] Mp4Rec rec stop !!!!!!!!!!!!!\n");
    return 0;
}

int Mp4Rec::write_one_frame(uint8_t *buf_h264,uint64_t size)
{
    if(!_start)
        return -1;
    if (MP4E_STATUS_OK != mp4_h26x_write_nal(&_mp4wr, buf_h264, size, 90000/_fps))
    {
        printf("error: mp4_h26x_write_nal failed\n");
        return -2;
    }
    return 0;
}













