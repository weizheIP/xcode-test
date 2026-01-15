#ifndef FACE_DB
#define FACE_DB

#include <cstdio>
#include <cstdlib>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#include <faiss/index_io.h>
#include <faiss/impl/IDSelector.h>

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <vector>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <dirent.h>
#include <mutex>


#include "common/MyLock/write_priotity_lock.h"

#include "post/queue.h"
// #include "post/model_base.h"
#include "post/inferModel/face_rec.h"



#include "json.h"


//using idx_t = faiss::idx_t;





struct faceInfo;


class FaceDb
{
public:
FaceDb(int d);
~FaceDb();
int getSize();
int init(std::vector<float> input);
int init2(std::string indexDbPath);
int add(int num,std::vector<float> input);
int del(int n, idx_t *arr);
int search(int k,int nq,std::vector<float> xq,std::vector<idx_t> &I,std::vector<float> &D);
int search2(int k,int nq,std::vector<float> xq,std::vector<idx_t> &I,std::vector<float> &D);
int searchforTool(int k,int nq,std::vector<float> xq,std::vector<idx_t> &I,std::vector<float> &D);

int dimension;
faiss::IndexFlatL2 *index;
private:
idx_t idIndex;
std::vector<idx_t> delIdArr;

bool isInit;

};


class FaceServer
{
public:
FaceServer(std::string inPath);
~FaceServer();


int GetIndexSize(){return idIndex;};

std::string getSearchResp(std::vector<idx_t> I,std::vector<float> D);

void server_th();

// int search(cv::Mat img,std::vector<faceInfo> &ans,std::vector<cv::Point> contours);

int search(std::vector<faceInfo> &ans,std::vector<Yolov5Face_BoxStruct> &face_result,std::vector<int> &delNums,std::vector<float> &input,int querySize,int kSize);
int searchForTool(std::vector<faceInfo> &ans,std::vector<Yolov5Face_BoxStruct> &face_result,std::vector<int> &delNums,std::vector<float> &input,int querySize, int kSize);

void save_mind_data_thread();
std::thread* th_ptr2;
BlockingQueue<int>* saveFlagQue;
std::vector<idx_t> spIdVec;
std::map<idx_t,std::string> spIdFileNameMap;
std::map<std::string,idx_t> fileNameSpIdMap;


private:
bool mStatus;
FaceDb *facedb;
std::string mImgsPath;
std::thread *th_ptr;
std::map<idx_t,std::string> fileIdMap;
std::map<std::string,idx_t> idFileMap;
idx_t idIndex;


write_priotity_lock dbLock;

std::mutex serverLock;

//faceRec facetool;
faceRec& facetool = faceRec::instance();

bool _reload;

};



#ifdef STRANGERDB
class StrangeServer
{
public:
    StrangeServer();
    ~StrangeServer();
    void server_th(); //search
    // int search_for_add(std::vector<float>& target,vector<float>& D,vector<idx_t>& I);
    string getSearchResp(float confff,vector<idx_t> I,vector<float> D);
    int add(std::vector<float>& target);
    
    void save_mind_data_thread();
    std::thread* th_ptr2;
    BlockingQueue<int>* saveFlagQue;
    std::vector<idx_t> spIdVec;
    
private:

bool mStatus;
FaceDb *facedb;
std::string mImgsPath;
std::thread *th_ptr;
std::map<idx_t,std::string> fileIdMap;
std::map<std::string,idx_t> idFileMap;
idx_t idIndex;
write_priotity_lock dbLock;
std::mutex serverLock;

faceRec facetool;
};

#endif



#endif