#include "httpprocess.h"


    // HttpProcess(string Ip);
    // ~HttpProcess();
    // void Get(string path,string& data);
    // void Post(string path,string& data);

HttpProcess::HttpProcess(string Ip):ip(Ip)
{

}

void HttpProcess::Post(string path,string& data)
{
  auto scheme_host_port = ip.c_str();
  //cout<<"send<<"<<ip.c_str()<<path.c_str()<<endl;
  if (auto res = httplib::Client(scheme_host_port).Post(path.c_str())) {
    // cout << res->status << endl;
    // cout << res->get_header_value("Content-Type") << endl;
    // cout << res->body << endl;
    data = res->body;
  } else {
    char buff[20]{0};
    snprintf(buff,sizeof(buff),"%d", (int)(res.error()));
    data = buff;
    
  }
}

void HttpProcess::Post_Json(string path,string str,string& request)
{
  auto scheme_host_port = ip.c_str();
  httplib::Params params;

  httplib::Client client(scheme_host_port);
  httplib::Client(scheme_host_port).set_read_timeout(3,0);
  httplib::Client(scheme_host_port).set_write_timeout(3,0);
  client.set_connection_timeout(1,0);
  client.set_read_timeout(1);
  client.set_write_timeout(1);
  if (auto res = client.Post(path.c_str(),str, "application/json; charset=utf-8")) {

    request = res->body;
    //cout<<"get req"<<request<<endl;
  } else {
    char buff[20]{0};
    snprintf(buff,sizeof(buff),"%d", (int)(res.error()));
    request = buff;
  }
}

void HttpProcess::Post(string path,string rtsp,string& request)
{
  auto scheme_host_port = ip.c_str();
  httplib::Params params;
  params.insert({"rtsp",rtsp});
  if (auto res = httplib::Client(scheme_host_port).Post(path.c_str(),params)) {
    // cout << "Post Succ" << endl;
    // cout << res->status << endl;
    // cout << res->get_header_value("Content-Type") << endl;
    // cout << res->body << endl;

    request = res->body;
  } else {
    // cout << "Post Error" << endl;
    // cout << res.error() << endl;
    char buff[20]{0};
    snprintf(buff,sizeof(buff),"%d", (int)(res.error()));
    request = buff;
  }
}

void HttpProcess::Get(string path,string& data)
{
  auto scheme_host_port = ip.c_str();

  httplib::Client client(scheme_host_port);
  // client.set_keep_alive(true);  
  // client.set_read_timeout(2);  //5s有风险,相机多的时候要14s
  // client.set_write_timeout(2);  
  client.set_connection_timeout(10,0);
  client.set_read_timeout(10);  //5s有风险,相机多的时候要14s
  client.set_write_timeout(10);  

  // client.set_connection_timeout(100,0);
  // client.set_read_timeout(100);  //5s有风险,相机多的时候要14s
  // client.set_write_timeout(100);  

  // if (auto res = httplib::Client(scheme_host_port).Get(path.c_str())) {
  if (auto res = client.Get(path.c_str())) {
    // cout << res->status << endl;
    // cout << res->get_header_value("Content-Type") << endl;
    // cout << res->body << endl;
    data = res->body;
  } else {
    cout << "[ERROR] http camera all: " << httplib::to_string(res.error()) << endl;
    // char buff[20]{0};
    // snprintf(buff,sizeof(buff),"%d", (int)(res.error()));
    data = httplib::to_string(res.error());
  }
}
Json::Value HttpProcess::Jsonparse(string data)
{
  Json::Reader reader;
  Json::Value value;
  reader.parse(data,value);
  return value;
}
HttpProcess::~HttpProcess(){
  // httplib::Client::stop()
}

    //   char* data="{\"123\":\"64\"}";
    // Json::Reader read;
    // Json::Value value;
    // read.parse(data,value);
    // auto d = value["123"].asString();


