#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <net/if.h>
#include <string.h>
#include "license.h"

int get_mac(char* & mac)
{

        struct ifreq ifreq;
        int sock;

        // if(argc!=2)
        // {
        //         printf("Usage : ethname\n");
        //         return 1;
        // }
        if((sock=socket(AF_INET,SOCK_STREAM,0))<0)
        {
                perror("socket");
                return -1;
        }
        strcpy(ifreq.ifr_name,"eth0");
        if(ioctl(sock,SIOCGIFHWADDR,&ifreq)<0)
        {
                perror("ioctl");
                return -2;
        }
        mac = new char[20]();
        snprintf(mac,20,"%02x:%02x:%02x:%02x:%02x:%02x",
                        (unsigned char)ifreq.ifr_hwaddr.sa_data[0],
                        (unsigned char)ifreq.ifr_hwaddr.sa_data[1],
                        (unsigned char)ifreq.ifr_hwaddr.sa_data[2],
                        (unsigned char)ifreq.ifr_hwaddr.sa_data[3],
                        (unsigned char)ifreq.ifr_hwaddr.sa_data[4],
                        (unsigned char)ifreq.ifr_hwaddr.sa_data[5]);
        // printf("%02x:%02x:%02x:%02x:%02x:%02x\n",
        //                 (unsigned char)ifreq.ifr_hwaddr.sa_data[0],
        //                 (unsigned char)ifreq.ifr_hwaddr.sa_data[1],
        //                 (unsigned char)ifreq.ifr_hwaddr.sa_data[2],
        //                 (unsigned char)ifreq.ifr_hwaddr.sa_data[3],
        //                 (unsigned char)ifreq.ifr_hwaddr.sa_data[4],
        //                 (unsigned char)ifreq.ifr_hwaddr.sa_data[5]);
        return 0;
}


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

// cat /sys/class/net/e*/address 网口不叫eth0的时候
// int cpu_serial2(std::string &serial){
// int cpu_eth0(std::string &serial){
//         FILE* fp = fopen("/sys/class/net/eth0/address","r");// 
//         if(fp<0)return -1;
//         fseek(fp,0,SEEK_END);
//         size_t size_f = ftell(fp);
//         fseek(fp,0,SEEK_SET);
//         char buffer[50]={0};
//         fread(buffer,size_f,1,fp);
//         fclose(fp);
//         serial = buffer;
//         int sindex = serial.find("\n");
//         serial = serial.substr(0,sindex);
//         // std::cout<<"serial:"<<serial<<std::endl;
//         return 0;
// }

int get_cpuSerial_and_Id(char* &serial)
{
        FILE* fp = popen("echo $(dmidecode -t 2 | grep Serial | sed 's/.*Serial Number://;s/ //g')-$(dmidecode -t 4 | grep ID | sed 's/.*ID://;s/ //g')","r");
        char buffer[100]={0};
        fread(buffer,100,1,fp);
        fclose(fp);
        buffer[strlen(buffer)-1]='\0';
        std::cout<<buffer<<std::endl;
        strcpy(serial,&(buffer[0]));
        return 0;
}

// int get_cpuSerial_and_Id2(std::string &serial)
// {
//         FILE* fp = popen("echo $(dmidecode -t 2 | grep Serial | sed 's/.*Serial Number://;s/ //g')-$(dmidecode -t 4 | grep ID | sed 's/.*ID://;s/ //g')","r");
//         char buffer[100]={0};
//         fread(buffer,100,1,fp);
//         fclose(fp);
//         buffer[strlen(buffer)-1]='\0';
//         // std::cout<<buffer<<std::endl;
//         // strcpy(serial,&(buffer[0]));
//         serial = buffer;
//         std::cout<<serial<<std::endl;
//         return 0;
// }

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
        std::cout<<"["<<serial<<"]"<<std::endl;
        return 0;
}


int default_serial(std::string &str){
        char serial[200]={0};
        // FILE* fp = popen("cat /sys/class/net/eth0/address","r");
        #ifdef ATLAS_210
        FILE* fp = popen("cat /sys/class/net/eth1/address","r");
        // #elif ATLAS_500PRO
	// //FILE* fp = popen("cat /sys/class/net/enp125s0f0/address","r");
	// FILE * fp = popen("cat /sys/class/net/bond0/address","r");
        #else
        FILE* fp = popen("cat /sys/class/net/eth0/address","r");
        #endif
        char buffer[100]={0};
        int ret = fread(buffer,100,1,fp);
        if(ret < 0)
                return -1;
        fclose(fp);
        buffer[strlen(buffer)-1]='\0';
        // strcpy(serial,&(buffer[10]));
        strcpy(serial,&(buffer[0]));
        str = serial;
        return 0;
}