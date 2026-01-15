#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <net/if.h>
#include <string.h>
#include "license.h"
#include "common/http/httpprocess.h"

int cpu_serial(char* &serial){
        FILE* fp = popen("cat /proc/cpuinfo | grep Serial","r");
        char buffer[100]={0};
        fread(buffer,100,1,fp);
        fclose(fp);
        buffer[strlen(buffer)-1]='\0';
        // strcpy(serial,&(buffer[10]));
        strcpy(serial,&(buffer[0]));
        return 0;
}
int cpu_serial2(std::string &str){
        FILE* fp = popen("cat /proc/cpuinfo | grep Serial","r");
        char buffer[100]={0};
        fread(buffer,100,1,fp);
        fclose(fp);
        buffer[strlen(buffer)-1]='\0';
        str = buffer;

        return 0;
}

int get_cpuSerial_and_Id(char* &serial)
{
        FILE* fp = popen("echo $(dmidecode -t 2 | grep Serial | sed 's/.*Serial Number://;s/ //g')-$(dmidecode -t 4 | grep ID | sed 's/.*ID://;s/ //g')","r");
        char buffer[100]={0};
        fread(buffer,100,1,fp);
        fclose(fp);
        buffer[strlen(buffer)-1]='\0';
        // std::cout<<buffer<<std::endl;
        strcpy(serial,&(buffer[0]));
        return 0;
}
int get_cpuSerial_and_Id2(std::string &serial)
{
        FILE* fp = popen("echo $(dmidecode -t 2 | grep Serial | sed 's/.*Serial Number://;s/ //g')-$(dmidecode -t 4 | grep ID | sed 's/.*ID://;s/ //g')","r");
        // char buffer[100]={0};
        char *buffer = (char *)malloc(1000);
        memset(buffer,0,1000);
        int index = 0;
        char c;
        while((c = fgetc(fp)) != EOF)
        {
                buffer[index] = c;
                index++;
        }
        buffer[index-1] = '\0';
        fclose(fp);

        serial = buffer;
        free(buffer);
        // std::cout<<"["<<serial<<"]"<<std::endl;
        return 0;
}


int default_serial(std::string &str){
        char serial[300]={0};
        // FILE* fp = popen("cat /sys/class/net/eth0/address","r");
        #ifdef ATLAS_210
        FILE* fp = popen("cat /sys/class/net/eth1/address","r");
        #elif defined(ATLAS_500PRO)
	FILE* fp = popen("cat /sys/class/net/e*/address","r");
	// FILE * fp = popen("cat /sys/class/net/bond0/address","r");
        #elif defined(ZHIWEI_GONGKONG)
        FILE* fp = popen("cat /sys/class/net/e*/address","r");
        #elif defined(JETSON808)
        FILE* fp = popen("grep -hroE '^[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}$' /sys/class/net/e*/address | head -n 1","r");        
        #else
        FILE* fp = popen("cat /sys/class/net/eth0/address","r");
        #endif
        char buffer[300]={0};
        int ret = fread(buffer,300,1,fp);
        if(ret < 0)
                return -1;
        fclose(fp);
        buffer[strlen(buffer)-1]='\0';
        // strcpy(serial,&(buffer[10]));
        strcpy(serial,&(buffer[0]));
        str = serial;
        return 0;
}



int http_serial(std::string &serial){
        HttpProcess http("http://127.0.0.1:9696");
        std::string data;
        while(1)
        {
        
                http.Get("/api/system/device",data);
                if(data.find("data")==std::string::npos) {
                        usleep(1000000);
                        std::cout<<"[ERROR] Get device failed!!!"<<std::endl;
                        continue;
                }
                else break;
        }
        Json::Value value = http.Jsonparse(data);
        
        if(value.isNull() || value["data"].isNull())
        {
                serial = "";
                return -1;
        }
        else{
                serial = value["data"].asString();
                return 0;
        }
}
