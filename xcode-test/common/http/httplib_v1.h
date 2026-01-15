#ifndef __HTTP_Lib_V1_H__
#define __HTTP_Lib_V1_H__

#include "httplib.h"
#include "json.h"
#include <iostream>

class httpLibV1
{
public:
    std::string ip;
    httplib::Client *_client = NULL;

    httpLibV1(std::string ip)
    {
        this->ip = ip;
        _client = new httplib::Client(ip);
    }
    ~httpLibV1()
    {
        if(_client)
            delete _client;
    }

    void Get(std::string path,std::string& data)
    {
        if(_client == NULL)
        {
            std::cout<<"http not init"<<"\n";
            return ;
        }
        if (auto res = _client->Get(path.c_str())) {
            // cout << res->status << endl;
            // cout << res->get_header_value("Content-Type") << endl;
            // cout << res->body << endl;
            data = res->body;
        } else {
            // cout << res.error() << endl;
            char buff[20]{0};
            snprintf(buff,sizeof(buff),"%d", (int)(res.error()));
            data = buff;
        }
    }

    void Get_One(std::string path,std::string& data)
    {
        
        if (auto res = httplib::Client(ip).Get(path.c_str())) {
            // cout << res->status << endl;
            // cout << res->get_header_value("Content-Type") << endl;
            // cout << res->body << endl;
            data = res->body;
        } else {
            // cout << res.error() << endl;
            char buff[20]{0};
            snprintf(buff,sizeof(buff),"%d", (int)(res.error()));
            data = buff;
            std::cout<<ip<<":"<<path<<"\n";
        }
    }

    Json::Value Jsonparse(std::string data)
    {
        Json::Reader reader;
        Json::Value value;
        reader.parse(data,value);
        return value;
    }
    
};

#endif