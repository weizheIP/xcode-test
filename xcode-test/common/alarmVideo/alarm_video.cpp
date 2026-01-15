#include "alarm_video.h"

int AlarmVideoInfo::switchAlarm = 0;

std::vector<std::shared_ptr<T_AlarmInfo>> AlarmVideoInfo::alarm_infos;
std::mutex AlarmVideoInfo::alarm_mutex;

int AlarmVideoInfo::sim_alarm_flag = 1; //模拟报警标志

std::unordered_map<std::string, std::condition_variable> AlarmVideoInfo::cam_cvs; // 按 camID 存储条件变量
std::unordered_map<std::string, bool> AlarmVideoInfo::cam_flags; 				 // 标志位，指示是否需要处理
std::unordered_map<std::string, bool> AlarmVideoInfo::cam_stops;                  // 标志位，指示是否停止某个cam

std::mutex AlarmVideoInfo::cam_mutex; 

std::string AlarmVideoInfo::alarm_path = "/civi/civiapp/resource";

std::mutex AlarmVideoInfo::log_mutex;


