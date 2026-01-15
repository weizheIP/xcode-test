#pragma once
#include <iostream>
// bool check_license_by_path(char* licpath);
bool check_license_by_file(const char* en_license_path);
void encode_license_to_path(char* licpath);
int get_model_decrypt_value(unsigned char* encrypt,unsigned char* modelvalue,long len_);
bool checklic(std::string sLicense_path);


