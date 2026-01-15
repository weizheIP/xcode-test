#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
// #include <unistd.h>
#include <string.h>
// #include <sys/socket.h>
// #include <netinet/in.h>
// #include <net/if.h>
// #include <netdb.h>
// #include <arpa/inet.h>
// #include <sys/ioctl.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include "license.h"
#include <map>
#include <stdio.h>
//#include <cpuid.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
//#include <asm/a.out.h>
#include <iostream>
// #include <file.hpp>
// #include <encryption.hpp>
#include <algorithm>
// #include <sys/time.h>
#include "plusaes.hpp"
#include "time.h"
#include "sys/time.h"
#include <thread>
#include "get_mac.h"
#include "license.h"
#include "base64.h"
#include <openssl/md5.h> 

typedef struct sdc_mmz_alloc_stru
{
	uint64_t addr_phy;
	uint64_t addr_virt;
	uint32_t size;
	uint32_t reserve;
	uint32_t cookie[4];
}sdc_mmz_alloc_s;
#define HI_VOID                 void

#define model_dir "model-0402-cow"
#define encry_mem len

#define __SERIAL__

extern std::string sTypeName;

char* get_SN()
{
    std::ifstream ifs("/civi/devicetree_base/serial-number", std::ios::binary | std::ios::in);
    if (!ifs.is_open()) {
    return 0;
    }
    ifs.seekg(0, std::ios::end);
    int size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::string str(size, '\0');
    ifs.read((char*)str.data(), size);
    return (char*)str.c_str();
}


int get_model_encrypt_value(unsigned char* encrypt,unsigned char* modelvalue,long len_,std::string outfilepath)
{
    long len = 10000*16;

    const std::vector<unsigned char> key = plusaes::key_from_string(&"civicint1110Encr"); // 16-char = 128-bit
    const unsigned char iv[16] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    };
    //encrypt
    // unsigned char* encrypted = (unsigned char*)malloc(len);
    // unsigned long en_len = len;
    // plusaes::encrypt_cbc(encrypt, len, &key[0], key.size(),&iv,encrypted,en_len,0);
    // decrypt
    modelvalue = (unsigned char*)malloc(len);


    struct timespec time1 = {0, 0}; 
    struct timespec time2 = {0, 0};

    clock_gettime(CLOCK_BOOTTIME, &time2);
    plusaes::encrypt_cbc(encrypt, len, &key[0], key.size(), &iv, modelvalue, len, 0);
    clock_gettime(CLOCK_BOOTTIME, &time1);

    memcpy(encrypt,modelvalue,len);
    free(modelvalue);
    char file_cbc[50];
    //snprintf(file_cbc,sizeof(file_cbc),"%s/model_cbc.wk",model_dir);
    // snprintf(file_cbc,sizeof(file_cbc),"m.pt",model_dir);
    FILE* output = fopen(outfilepath.c_str(),"w");
    fwrite(encrypt,len_,1,output);
    fclose(output);
    // Hello, plusaes
    // std::string tout( decrypted.begin(), decrypted.end() );
    // std::cout << tout << std::endl;
    return 0;
}


int get_model_decrypt_value(unsigned char* encrypt,unsigned char* modelvalue,long len_)
{
    long len = 10000*16;

    const std::vector<unsigned char> key = plusaes::key_from_string(&"civicint1110Encr"); // 16-char = 128-bit
    const unsigned char iv[16] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    };
    //encrypt
    // unsigned char* encrypted = (unsigned char*)malloc(len);
    // unsigned long en_len = len;
    // plusaes::encrypt_cbc(encrypt, len, &key[0], key.size(),&iv,encrypted,en_len,0);
    // decrypt
    modelvalue = (unsigned char*)malloc(len);


    struct timespec time1 = {0, 0}; 
    struct timespec time2 = {0, 0};

    clock_gettime(CLOCK_BOOTTIME, &time2);
    plusaes::decrypt_cbc(encrypt, len, &key[0], key.size(), &iv, modelvalue, len, 0);
    clock_gettime(CLOCK_BOOTTIME, &time1);

    memcpy(encrypt,modelvalue,len);

    free(modelvalue);

    // char file_cbc[50];
    // snprintf(file_cbc,sizeof(file_cbc),"%s/model_de_cbc.wk",model_dir);
    // FILE* output = fopen(file_cbc,"w");
    // fwrite(encrypt,len_,1,output);
    // fclose(output);
    // Hello, plusaes
    // std::string tout( decrypted.begin(), decrypted.end() );
    // std::cout << tout << std::endl;
    return 0;
}


unsigned char* encode(unsigned char* & data,int len)
{

    std::vector<unsigned char> kk;
    kk.push_back(0x11);
    const std::vector<unsigned char> key = plusaes::key_from_string(&"civicint1110Encr"); // 16-char = 128-bit
    const unsigned char iv[16] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    };
    //encrypt
    // unsigned char* encrypted = (unsigned char*)malloc(len);
    // unsigned long en_len = len;
    // plusaes::encrypt_cbc(encrypt, len, &key[0], key.size(),&iv,encrypted,en_len,0);
    // decrypt

    struct timespec time1 = {0, 0}; 
    struct timespec time2 = {0, 0};


    unsigned char* output = (unsigned char*)malloc(len*sizeof(unsigned char));
    plusaes::encrypt_cbc(data, len, &key[0], key.size(), &iv, output, len, 0);
    return output;


}
unsigned char* decode(unsigned char* encrypt,int len)
{
    const std::vector<unsigned char> key = plusaes::key_from_string(&"civicint1110Encr"); // 16-char = 128-bit
    const unsigned char iv[16] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    };
    //encrypt
    // unsigned char* encrypted = (unsigned char*)malloc(len);
    // unsigned long en_len = len;
    // plusaes::encrypt_cbc(encrypt, len, &key[0], key.size(),&iv,encrypted,en_len,0);
    // decrypt
    unsigned char* modelvalue = (unsigned char*)malloc(len);


    struct timespec time1 = {0, 0}; 
    struct timespec time2 = {0, 0};

    clock_gettime(CLOCK_BOOTTIME, &time2);
    plusaes::decrypt_cbc(encrypt, len, &key[0], key.size(), &iv, modelvalue, len, 0);
    clock_gettime(CLOCK_BOOTTIME, &time1);

    //  char file_cbc[50];
    // snprintf(file_cbc,sizeof(file_cbc),"%s/model_de_cbc.wk",model_dir);
    // FILE* output = fopen(file_cbc,"w");
    // fwrite(encrypt,len_,1,output);
    // fclose(output);
    // Hello, plusaes
    // std::string tout( decrypted.begin(), decrypted.end() );
    // std::cout << tout << std::endl;

    return modelvalue;
}

#if 1
extern std::vector<std::string> lic_algs;
std::string device_id;
bool check_license_by_file(const char* en_license_path)
{
    char *pMac = new char[512]();
    
    #if defined(__MAC__)
    get_mac(pMac);
    #endif

    #if defined(__SN__)
    pMac = get_SN();
    #endif

    #if defined(__SERIAL__)
    // cpu_serial(pMac);
    get_cpuSerial_and_Id(pMac);
    #endif
    //printf("i in %s\n",pMac);
    if(pMac=="")return false;
    device_id = pMac;
    // printf(pMac);
    // printf("\n");
    
    // std::ifstream in(en_license_path);    
    // if (!in.is_open())
    // {
    //     std::cerr << "open file failed!" << std::endl;
    //     return false;
                
    //     // exit(-1);
    // }
    // std::string s = "";
    // std::getline(in,s);  
    // in.close();
    if(!(access(en_license_path, 0) == 0))
    {
        std::cout<<"lic is not exist!"<<std::endl;
        return false;
    }
    FILE *fp = fopen(en_license_path,"r");
    char file_buf[512];
    if (fseek(fp, 0, SEEK_END) != 0) {	// 移动文件指针到文件末尾
		printf("fseek failed: %s\n", strerror(errno));
		return -1;
	}
	int file_size = ftell(fp);	// 获取此时偏移值，即文件大小
	if (file_size == -1) {
		printf("ftell failed :%s\n", strerror(errno));
	}
	if (fseek(fp, 0, SEEK_SET) != 0) {	// 将文件指针恢复初始位置
		printf("fseek failed: %s\n", strerror(errno));
		return -1;
	}
    fread(file_buf,1,file_size,fp);
    
    std::string s = file_buf;
    // std::cout << s << std::endl;        
    //std::vector<unsigned char> encrypted( s.begin(),s.end() );
    std::vector<unsigned char> encrypted(file_buf,file_buf+file_size);
    // std::string t3( encrypted.begin(), encrypted.end() );
    //std::cout << s << std::endl;

    const std::vector<unsigned char> key = plusaes::key_from_string(&"civicint1110Encr"); // 16-char = 128-bit
    const unsigned char iv[16] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    };
    // decrypt
    unsigned long padded_size = 0;
    std::vector<unsigned char> decrypted(encrypted.size());

    plusaes::decrypt_cbc(&encrypted[0], encrypted.size(), &key[0], key.size(), &iv, &decrypted[0], decrypted.size(), &padded_size);
    // Hello, plusaes
    std::string tout( decrypted.begin(), decrypted.end() );


    bool rr=true;
    std::string pp;
    #if defined(__SN__)
    pp = get_SN();
    #endif
    #if defined(__MAC__)
    get_mac(pMac);
    pp = pMac;
    #endif
    #if defined(__SERIAL__)
    cpu_serial(pMac);
    pp = pMac;
    #endif
    //std::cout << "decode lic:"<<tout<<std::endl;
    //std::cout << "device info:"<<pp<<std::endl;
    //类别验证
    const char *tempStr = tout.c_str();
    char buf[400] = {0};
    memccpy(buf,tempStr,1,strlen(tempStr)+1);
    //std::cout << buf << std::endl;
    const char *Schar ="&&";
    char *item;
    std::vector<std::string> strArr; 
    item = strtok(buf, Schar);
    tout = item;
    //std::cout << tout << std::endl;
    while( 1 ) {
      item = strtok(NULL,Schar);
      if(item == NULL)
        break;
      std::string temp = item;
      //std::cout << temp << std::endl;
      strArr.push_back(temp);
   }
   std::cout<<"Authorized algorithm:"<<"\n";
   for(int i=0;i<strArr.size();i++)
   {
        // if(strArr[i] == sTypeName)
        //     break;
        // if(i == strArr.size()-1)
        // {
        //     printf("lic classTpye not right!\n");
        //     return false;
        // }
        lic_algs.push_back(strArr[i]);
        std::cout<<strArr[i]<<"\n";
   }

    if (tout.substr(0,pp.length())==pp.substr(0,pp.length())){    
        rr=true;
        //printf("111");
    }else{
      //printf("222");
        rr=false;
    }
    // free(pMac);
    delete[] pMac;
    return rr;
}
#endif


bool check_license_by_path(char* licpath)
{
    std::ifstream in(licpath);    
    if (!in.is_open())
    {
        return false;
        // std::cerr << "open file failed!" << std::endl;        
        // exit(-1);
    }
    std::string s = "";
    std::getline(in,s);  
    in.close();    
    std::vector<unsigned char> fo(s.begin(),s.end());
    unsigned char* fo2 = decode(&(fo[0]),s.length());

    //char* mac_msg;
    //get_mac(mac_msg);
    char* mac_msg = new char[320]();
    #if defined(__SN__)
    mac_msg = get_SN();
    #endif
    #if defined(__MAC__)
    get_mac(mac_msg);
    #endif
    #if defined(__SERIAL__)
    // cpu_serial(mac_msg);
    get_cpuSerial_and_Id(mac_msg);
    #endif
    

    if(strstr((char*)fo2,mac_msg))
    {
        return 1;
    }
    else return 0;

}

std::vector<std::string> g_typeArr;
void encode_license_to_path(std::string ID, char* licpath)
{
    //char* mac_msg;
    //get_mac(mac_msg);
    char* mac_msg = new char[512]();
    #if defined(__SN__)
    mac_msg = get_SN();
    #endif
    #if defined(__MAC__)
    get_mac(mac_msg);
    #endif
    #if defined(__SERIAL__)
    // cpu_serial(mac_msg);
    // get_cpuSerial_and_Id(mac_msg);
    memcpy(mac_msg,ID.c_str(),ID.size());
    #endif

    #if 1
    //加算法类别 &&进行分割 cpuInfo&&type1&&type2
    int licStrlen = strlen(mac_msg);
    printf("%d\n",g_typeArr.size());
    for(int i = 0; i < g_typeArr.size() ;i++)
    {
        mac_msg[licStrlen++] = '&';
        mac_msg[licStrlen++] = '&';
        for(int j=0; j<g_typeArr[i].size(); j++)
        {
            mac_msg[licStrlen++] = g_typeArr[i][j];
        }
    }
    mac_msg[licStrlen] = '\0';
    #endif

    printf("info:%s\n",mac_msg);
    std::string mac_str = mac_msg; 
    // unsigned char* mac_uchar = (unsigned char*)mac_msg;
    int ic = strlen(mac_msg);
    unsigned char* mac_uchar = new unsigned char[512]();
     
    memset(mac_uchar,0,512);
    memcpy(mac_uchar,mac_msg,ic);

    unsigned char* o = encode(mac_uchar,512);

    delete[] mac_uchar;
    delete[] mac_msg;
    FILE* f = fopen(licpath,"wb");
    fwrite(o,512,1,f);
    fclose(f);

}

int get_license_stypes(const char* en_license_path)
{
    char *pMac = new char[256]();
    
    #if defined(__MAC__)
    get_mac(pMac);
    #endif

    #if defined(__SN__)
    pMac = get_SN();
    #endif

    #if defined(__SERIAL__)
    cpu_serial(pMac);
    #endif
    //printf("i in %s\n",pMac);
    if(pMac=="")return false;
    // printf(pMac);
    // printf("\n");
    
    // std::ifstream in(en_license_path);    
    // if (!in.is_open())
    // {
    //     std::cerr << "open file failed!" << std::endl;
    //     return false;
                
    //     // exit(-1);
    // }
    // std::string s = "";
    // std::getline(in,s);  
    // in.close();
    FILE *fp = fopen(en_license_path,"r");
    
    if (fseek(fp, 0, SEEK_END) != 0) {	// 移动文件指针到文件末尾
		printf("fseek failed: %s\n", strerror(errno));
		return -1;
	}
	int file_size = ftell(fp);	// 获取此时偏移值，即文件大小
	if (file_size == -1) {
		printf("ftell failed :%s\n", strerror(errno));
	}
	if (fseek(fp, 0, SEEK_SET) != 0) {	// 将文件指针恢复初始位置
		printf("fseek failed: %s\n", strerror(errno));
		return -1;
	}
    char* file_buf = (char*)malloc(file_size);
    fread(file_buf,1,file_size,fp);
    
    std::string s = file_buf;
    // std::cout << s << std::endl;        
    //std::vector<unsigned char> encrypted( s.begin(),s.end() );
    std::vector<unsigned char> encrypted(file_buf,file_buf+file_size);
    // std::string t3( encrypted.begin(), encrypted.end() );
    //std::cout << s << std::endl;

    const std::vector<unsigned char> key = plusaes::key_from_string(&"civicint1110Encr"); // 16-char = 128-bit
    const unsigned char iv[16] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    };
    // decrypt
    unsigned long padded_size = 0;
    std::vector<unsigned char> decrypted(encrypted.size());

    plusaes::decrypt_cbc(&encrypted[0], encrypted.size(), &key[0], key.size(), &iv, &decrypted[0], decrypted.size(), &padded_size);
    // Hello, plusaes
    std::string tout( decrypted.begin(), decrypted.end() );


    bool rr=true;
    std::string pp;
    #if defined(__SN__)
    pp = get_SN();
    #endif
    #if defined(__MAC__)
    get_mac(pMac);
    pp = pMac;
    #endif
    #if defined(__SERIAL__)
    cpu_serial(pMac);
    pp = pMac;
    #endif
    if(strlen(pMac)==0){
        return false;
    }
    //std::cout << "decode lic:"<<tout<<std::endl;
    //std::cout << "device info:"<<pp<<std::endl;
    //类别验证
    const char *tempStr = tout.c_str();
    char buf[1000] = {0};
    memccpy(buf,tempStr,1,strlen(tempStr)+1);
    std::cout << buf << std::endl;
    const char *Schar ="&&";
    char *item;
    // std::vector<std::string> strArr; 
    item = strtok(buf, Schar);
    tout = item;
    //std::cout << tout << std::endl;
    while( 1 ) {
      item = strtok(NULL,Schar);
      if(item == NULL)
        break;
      std::string temp = item;
    //   std::cout << temp << std::endl;
      g_typeArr.push_back(temp);
   }
   lic_algs = g_typeArr;
   return g_typeArr.size();
}

void Change_char2charlist(char* input,unsigned char *iv){
    int len = strlen(input);
    for(int i = 0;i<len;i++)
    {
        // 48-57
        if(input[i]>=48 && input[i]<=57){
            iv[i] = input[i]-48;
        }
        else{// 97 - 
            iv[i] = input[i]-97+10;
        }
    }
}

unsigned char* decrypt_256_value(unsigned char* encrypt,long len_)
{

    const std::vector<unsigned char> key = plusaes::key_from_string(&"967583ac7ca31ff35845eef5a6fe523f"); // 16-char = 128-bit
    // const unsigned char iv[16] = { //
    //     0x0e, 0x9a, 0x0b, 0xdf, 0x99, 0x58, 0x06, 0x07,
    //     0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    // };
    unsigned char iv[16]={0};
    Change_char2charlist("ea9a0bdf9958d7bf",iv);

    //encrypt
    // unsigned char* encrypted = (unsigned char*)malloc(len);
    // unsigned long en_len = len;
    // plusaes::encrypt_cbc(encrypt, len, &key[0], key.size(),&iv,encrypted,en_len,0);
    // decrypt
    unsigned char* modelvalue = (unsigned char*)malloc(len_);


    struct timespec time1 = {0, 0}; 
    struct timespec time2 = {0, 0};

    clock_gettime(CLOCK_BOOTTIME, &time2);
    plusaes::decrypt_cbc(encrypt, len_, &key[0], key.size(), &iv, modelvalue, len_, 0);
    clock_gettime(CLOCK_BOOTTIME, &time1);

    for(int i =0 ;i<len_;i++)
    {
        if(modelvalue[i]=='\0')modelvalue[i]=' ';
        // std::cout<<modelvalue[i];
    }
    // std::cout<<std::endl;
    // memcpy(encrypt,modelvalue,len_);
    // std::string output = (char*)modelvalue;

    return modelvalue;

}

extern std::string lic_algs_all;
bool getstypearr(std::string strs,std::string sign=",",std::string stypename = sTypeName){
    std::string serial = "asdasdasdasd";
    std::vector<std::string> sts;
    // if(cpu_serial2(serial)!=0)
    //     return false;
    
    if(get_cpuSerial_and_Id2(serial)!=0)
        return false;
    // std::string device = base64_encode((unsigned char*)serial.c_str(),serial.size());
    MD5_CTX ctx;
    const char *data=serial.c_str();;
    unsigned char md[16]={0};
    char buf[33]={0};
    char tmp[3]={0};
    int i;

    MD5_Init(&ctx);
    MD5_Update(&ctx,data,strlen(data));
    MD5_Final(md,&ctx); 

    for( i=0; i<16; i++ ){
        sprintf(tmp,"%02X",md[i]);
        strcat(buf,tmp);
    }
    device_id = buf;
    // printf("%s\n",buf); 
    
    std::string tmp_str;
    int index_device = strs.find("device:B-");
    tmp_str = strs.substr(index_device,strs.size()-index_device);
    int index_ret = tmp_str.find("\n");
    tmp_str = strs.substr(index_device,index_ret);
    bool bDevice = strstr(tmp_str.c_str(),buf);

    int index_args = strs.find("algs:");
    // tmp_str = strs.substr(index_args,strs.size()-index_args);
    // bool bStype = strstr(tmp_str.c_str(),stypename.c_str());
    
    tmp_str = strs.substr(index_args+5,strs.size()-index_args);
    bool bStype = true;

    int oldIndex = 0;
    int newIndex = 0;
    std::vector<std::string> lic_algs2;
    for(int i = 0; i< tmp_str.size();i++)
    {
        if(tmp_str[i] == ',')
        {
            newIndex = i;
            lic_algs2.push_back(tmp_str.substr(oldIndex,oldIndex ? newIndex - oldIndex : newIndex));
            oldIndex = i+1;
        }
        if(i == tmp_str.size() -1)
        {
            lic_algs2.push_back(tmp_str.substr(oldIndex));
        }
    }
    
    lic_algs_all = tmp_str;

    lic_algs = lic_algs2;
    std::cout<<"------------------"<<std::endl;
    for(auto item : lic_algs2)
        std::cout<<item<<std::endl;
    std::cout<<"------------------"<<std::endl;


    if(bDevice && bStype){
        return true;
    }
    return false;
}

#define CODEC_ALIGN(x, a)   (((x)+(a)-1)&~((a)-1))

bool checklic(std::string sLicense_path)
{
    if(!access(sLicense_path.c_str(),NULL)){
        FILE* f = fopen(sLicense_path.c_str(),"r");
        fseek(f,0,SEEK_END);
        long size_f = ftell(f);
        long readsize = size_f;
        // if(size_f%64>0)
        //     size_f+=(64-(size_f%64));

        size_f++;//给末尾加\0,整体64位对齐
        size_f = CODEC_ALIGN(size_f,64); 

        fseek(f,0,SEEK_SET);

        int mallsize = size_f*sizeof(char);

        char* buffer= (char*)malloc(mallsize);
        
        memset(buffer,0,mallsize);
        
        fread(buffer,1,readsize,f);//防止越界

        buffer[mallsize - 1] = '\0'; // 读出来东西不算字符串，写入的时候没有写\0
        // printf("[%s]\n",buffer);
        fclose(f);

        std::string data_d = buffer;
        free(buffer);
        std::string d_data = base64_decode(data_d);
        unsigned char* dcr = decrypt_256_value((unsigned char*)d_data.c_str(),d_data.size());
        // std::cout<<"dcr:"<<dcr<<std::endl;
        return getstypearr(std::string((char*)dcr));
    }
    else 
    {
        std::cout<<"[Error] Could not find license file!"<<std::endl;
        return false;
    }
}




// int main(int argc, char** argv)
// {
//     std::string ID = argv[1];

//     if(ID == "")
//     {
//         std::cout<<"device Id is empty!"<<"\n";
//     }
    
//     for(int i = 2; i < argc ;i++)
//     {
//         g_typeArr.push_back(argv[i]);
//     } 
//     encode_license_to_path(ID,"./license.lic");
//     // bool check_lic = check_license_by_path("./license.lic");

//     return 0;
// }

// int main(int argc, char** argv)
// {
    
//     std::string file_path = argv[1];
    
//     if(file_path.find("m.pt")== std::string::npos)
//     {
//         std::cout<<"model file path not right."<<std::endl;
//         return -1;
//     }
//     int index_str = file_path.find("m.pt");
//     std::string outfilePath;
//     if(index_str == 0)
//     {
//         outfilePath = "m_enc.pt";
//     }
//     else
//     {
//         outfilePath = file_path.substr(0,index_str);
//         outfilePath.append("m_enc.pt");
//     }
    
//     //std::string file_pat = "m.pt";
//     FILE *ifp;
//     unsigned char *data;
//     unsigned char *outdata;

//     ifp = fopen(file_path.c_str(), "rb");//只读
//     if (NULL == ifp)
//     {
//         printf("model file dont exist.\n");
//         return NULL;
//     }
//     fseek(ifp, 0, SEEK_END);
//     int size = ftell(ifp);

//     int ret;

//     ret = fseek(ifp, 0, SEEK_SET);
//     if (ret != 0)
//     {
//         printf("blob seek failure.\n");
//         return NULL;
//     }

//     data = (unsigned char *)malloc(size);
//     if (data == NULL)
//     {
//         printf("buffer malloc failure.\n");
//         return NULL;
//     }
//     ret = fread(data, 1, size, ifp);
//     fclose(ifp);

//     get_model_encrypt_value(data,outdata,size,outfilePath);

//     return 0;
// }



