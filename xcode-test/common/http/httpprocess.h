#ifndef __HTTP_PROCESS_H__
#define __HTTP_PROCESS_H__

#include "httplib.h"
#include "json.h"
#include <iostream>

using namespace std;
class HttpProcess{
public:
    HttpProcess(string Ip);
    ~HttpProcess();
    void Get(string path,string& data);
    void Post(string path,string& data);
    void Post(string path,string rtsp,string& request);
    Json::Value Jsonparse(string data);
    void Post_Json(string path,string str,string& request);
    
    string ip;
};
#endif
