#ifndef ALARM_VIDEO_H
#define ALARM_VIDEO_H

#include <vector>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <algorithm>
#include <string>
#include <atomic>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <iomanip>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <ctime>

#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <errno.h>





#define OPEN_ALARM_VIDEO true

enum ALARM_STATUS {
    NEED_HANDLER = 0,
    HANDLERING,
    HANDLERED
};

struct T_AlarmInfo {
    std::string rtsp_path;
    std::string algo_name;
//    int camID;
    ALARM_STATUS alarm_status;
    std::string alarm_file_name;
};
using pt_AlarmInfo = T_AlarmInfo*;

class AlarmVideoInfo {
public:
	static int switchAlarm;
    static std::vector<std::shared_ptr<T_AlarmInfo>> alarm_infos;
    static std::mutex alarm_mutex;
	
    static int sim_alarm_flag; //模拟报警标志

	static std::unordered_map<std::string, std::condition_variable> cam_cvs; // 按 camID 存储条件变量
	static std::unordered_map<std::string, bool> cam_flags;                  // 标志位，指示是否需要处理
	static std::unordered_map<std::string, bool> cam_stops;                  // 标志位，指示是否需要处理
	static std::mutex cam_mutex; 

	static std::string alarm_path;

    // ====== 日志对外 API ======
    static void Log(const std::string& level, const std::string& msg) {
        WriteLog(level, /*rtsp_path*/"", /*alg*/"", msg);
    }

    static void Log(const std::string& level,
                    const std::string& rtsp_path,
                    const std::string& alg_name,
                    const std::string& msg) {
        WriteLog(level, rtsp_path, alg_name, msg);
    }

    // 便捷宏（在 .h 里也可）
#define LOG_DEBUG(msg) AlarmVideoInfo::Log("DEBUG", (msg))
#define LOG_INFO(msg)  AlarmVideoInfo::Log("INFO" , (msg))
#define LOG_ERROR(msg) AlarmVideoInfo::Log("ERROR", (msg))
#define LOG_WARN(msg) AlarmVideoInfo::Log("WARN", (msg))


#define LOG_DEBUG_RTSP(rtsp, alg, msg) AlarmVideoInfo::Log("DEBUG", (rtsp), (alg), (msg))
#define LOG_INFO_RTSP(rtsp, alg, msg)  AlarmVideoInfo::Log("INFO" , (rtsp), (alg), (msg))
#define LOG_ERROR_RTSP(rtsp, alg, msg) AlarmVideoInfo::Log("ERROR", (rtsp), (alg), (msg))
#define LOG_WARN_RTSP(rtsp, alg, msg) AlarmVideoInfo::Log("WARN", (rtsp), (alg), (msg))




    // Thread-safe addition of AlarmInfo
    static void AddAlarmInfo(std::shared_ptr<T_AlarmInfo> _info) 
    {
		if (!OPEN_ALARM_VIDEO)
			return;

		// 管理处理队列
        {
            std::lock_guard<std::mutex> lock(alarm_mutex);
//            std::cout << "[INFO] == Add alarm info from algo: " << _info->algo_name << std::endl;
			
			// Clean up HANDLERED items and find our target
			alarm_infos.erase(
				std::remove_if(alarm_infos.begin(), alarm_infos.end(),
					[](const std::shared_ptr<T_AlarmInfo>& info) { return info->alarm_status == HANDLERED; }),
				alarm_infos.end());

			// Add NEED_HANDLE INFO
			_info->alarm_status = NEED_HANDLER;			
            alarm_infos.push_back(_info);
        }

		// 通知某个摄像头
    	{
			std::lock_guard<std::mutex> lock(cam_mutex);
			cam_flags[_info->rtsp_path] = true;
			cam_stops[_info->rtsp_path] = false;
			cam_cvs[_info->rtsp_path].notify_one();
		}
    }

	static bool WaitAlarmInfoUpdate(const std::string& str_cam_id) 
	{

		std::unique_lock<std::mutex> lock(alarm_mutex);
		cam_cvs[str_cam_id].wait(lock, 
			[&] { return cam_flags[str_cam_id] || cam_stops[str_cam_id]; });
		if (cam_stops[str_cam_id]) {
			cam_stops[str_cam_id] = false;
			cam_flags[str_cam_id] = false;
			return false;
		}
		cam_flags[str_cam_id] = false;
		if (alarm_infos.empty()){
			std::cerr << "[ERROR] Alarm info list is empty for " << str_cam_id << std::endl;
			return false;
		}
			
		return true;
	}

	static void StopAlarmInfo(const std::string& str_cam_id)
	{
		std::unique_lock<std::mutex> lock(alarm_mutex);
		cam_stops[str_cam_id] = true;
		cam_cvs[str_cam_id].notify_one();
	}

	static int getCameraIdFromRTSP(const std::string& rtsp_url) {
	    // 1. 查找 '@' 符号（用户名密码后的IP起始位置）
	    size_t at_pos = rtsp_url.find('@');
	    if (at_pos == std::string::npos) {
	        throw std::runtime_error("Invalid RTSP URL format!");
	    }

	    // 2. 提取 IP:PORT 部分
	    std::string ip_part = rtsp_url.substr(at_pos + 1);
	    size_t colon_pos = ip_part.find(':');
	    size_t slash_pos = ip_part.find('/');
	    size_t ip_end_pos = (colon_pos != std::string::npos) ? colon_pos : slash_pos;
	    std::string ip_addr = ip_part.substr(0, ip_end_pos);

	    // 3. 提取最后一个 '.' 后的数字
	    size_t last_dot_pos = ip_addr.rfind('.');
	    if (last_dot_pos == std::string::npos) {
	        throw std::runtime_error("Invalid IP address format!");
	    }

	    std::string last_octet_str = ip_addr.substr(last_dot_pos + 1);
	    int camera_id;
	    std::istringstream iss(last_octet_str);
	    iss >> camera_id;

	    return camera_id;
	}
	
	// 提取通道号（优先匹配 "ch" 或 "/Channels/" 后的数字，否则返回 "00"）
	static std::string extractChannelNumber(const std::string& rtsp_url) {
		// 默认值
		std::string default_channel = "00";
	
		// 1. 查找 "/Channels/" 后的数字
		size_t channels_pos = rtsp_url.find("/Channels/");
		if (channels_pos != std::string::npos) {
			size_t start_pos = channels_pos + 10; // "/Channels/" 长度是 10
			size_t end_pos = rtsp_url.find_first_not_of("0123456789", start_pos);
			if (end_pos == std::string::npos) {
				end_pos = rtsp_url.length();
			}
			std::string channel = rtsp_url.substr(start_pos, end_pos - start_pos);
			if (!channel.empty()) {
				return channel;
			}
		}
	
		// 2. 查找 "ch" 后的数字（不区分大小写）
		size_t ch_pos = rtsp_url.find("ch");
		if (ch_pos == std::string::npos) {
			ch_pos = rtsp_url.find("CH"); // 尝试大写
		}
		if (ch_pos != std::string::npos) {
			size_t start_pos = ch_pos + 2; // "ch" 长度是 2
			size_t end_pos = rtsp_url.find_first_not_of("0123456789", start_pos);
			if (end_pos == std::string::npos) {
				end_pos = rtsp_url.length();
			}
			std::string channel = rtsp_url.substr(start_pos, end_pos - start_pos);
			if (!channel.empty()) {
				return channel;
			}
		}
	
		// 3. 如果未找到，返回默认值
		return default_channel;
	}

//	static std::string CreateVideoFileName(const std::string& rtsp_path, const std::string& alg_name)
//	{
//		// Get current time using C++11 chrono (thread-safe)
//		auto now = std::chrono::system_clock::now();
//		auto now_time_t = std::chrono::system_clock::to_time_t(now);
//		auto now_tm = *std::localtime(&now_time_t);  // This is still not thread-safe, 
//													// but better than storing pointer
//		
//		// Use stringstream for safer string formatting
//		std::ostringstream filename;
//		
//		// Format the filename
//		filename << alarm_path << "/"
//				 << std::setfill('0') 
//				 << (now_tm.tm_year + 1900) << "-"
//				 << std::setw(2) << (now_tm.tm_mon + 1) << "-"
//				 << std::setw(2) << now_tm.tm_mday << "/ip_"
//				 << AlarmVideoInfo::getCameraIdFromRTSP(rtsp_path) << "_ch"
//				 << AlarmVideoInfo::extractChannelNumber(rtsp_path) << "_alarm_"
//				 << alg_name << "_"
//				 << (now_tm.tm_year + 1900)
//				 << std::setw(2) << (now_tm.tm_mon + 1)
//				 << std::setw(2) << now_tm.tm_mday << "_"
//				 << std::setw(2) << now_tm.tm_hour
//				 << std::setw(2) << now_tm.tm_min
//				 << std::setw(2) << now_tm.tm_sec << ".mp4";
//		
//		return filename.str();
//	}

	static std::string CreateVideoFileName(const std::string& rtsp_path,
										   const std::string& alg_name) {
		// Get current time
		auto now = std::chrono::system_clock::now();
		auto now_time_t = std::chrono::system_clock::to_time_t(now);
		std::tm now_tm;
		localtime_r(&now_time_t, &now_tm);	// thread-safe
	
		// Ensure base alarm_path exists
		struct stat st_root;
		if (stat(alarm_path.c_str(), &st_root) != 0) {
			if (mkdir(alarm_path.c_str(), 0755) != 0 && errno != EEXIST) {
				std::cerr << "Failed to create base directory '" << alarm_path
						  << "': " << std::strerror(errno) << std::endl;
			}
		} else if (!S_ISDIR(st_root.st_mode)) {
			std::cerr << "Path '" << alarm_path << "' exists but is not a directory." << std::endl;
		}
	
		// Build date subdirectory: YYYY-MM-DD
		std::ostringstream date_ss;
		date_ss << std::setfill('0') 
				 << (now_tm.tm_year + 1900) << "-"
				 << std::setw(2) << (now_tm.tm_mon + 1) << "-"
				 << std::setw(2) << now_tm.tm_mday;
		const std::string date_dir = alarm_path + "/" + date_ss.str();
	
		// Check if date directory exists; if not, create it
		struct stat st;
		if (stat(date_dir.c_str(), &st) != 0) {
			if (mkdir(date_dir.c_str(), 0755) != 0 && errno != EEXIST) {
				std::cerr << "Failed to create directory '" << date_dir
						  << "': " << std::strerror(errno) << std::endl;
			}
		} else if (!S_ISDIR(st.st_mode)) {
			std::cerr << "Path '" << date_dir << "' exists but is not a directory." << std::endl;
		}
	
		// Build full filename with timestamp (use zero-pad to avoid spaces)
		std::ostringstream filename_ss;
		filename_ss  << alarm_path << "/"
					 << std::setfill('0') 
					 << (now_tm.tm_year + 1900) << "-"
					 << std::setw(2) << (now_tm.tm_mon + 1) << "-"
					 << std::setw(2) << now_tm.tm_mday << "/ip_"
					 << AlarmVideoInfo::getCameraIdFromRTSP(rtsp_path) << "_ch"
					 << AlarmVideoInfo::extractChannelNumber(rtsp_path) << "_alarm_"
					 << alg_name << "_"
					 << (now_tm.tm_year + 1900)
					 << std::setw(2) << (now_tm.tm_mon + 1)
					 << std::setw(2) << now_tm.tm_mday << "_"
					 << std::setw(2) << now_tm.tm_hour
					 << std::setw(2) << now_tm.tm_min
					 << std::setw(2) << now_tm.tm_sec << ".mp4";
	
		return filename_ss.str();
	}

private:
   // ====== 日志实现 ======
   static std::mutex log_mutex;

   // 返回并保证存在的当天目录：alarm_path/YYYY-MM-DD
   static std::string GetOrMakeTodayDir(std::tm& now_tm) {
	   // 保底：确保 alarm_path 存在
	   struct stat st_root;
	   if (stat(alarm_path.c_str(), &st_root) != 0) {
		   if (mkdir(alarm_path.c_str(), 0755) != 0 && errno != EEXIST) {
			   // 不能写日志就输出到stderr
			   std::cerr << "Failed to create base directory '" << alarm_path
						 << "': " << std::strerror(errno) << std::endl;
		   }
	   }

	   std::ostringstream date_ss;
	   date_ss << std::setfill('0')
			   << (now_tm.tm_year + 1900) << "-"
			   << std::setw(2) << (now_tm.tm_mon + 1) << "-"
			   << std::setw(2) << now_tm.tm_mday;
	   const std::string date_dir = alarm_path + "/" + date_ss.str();

	   struct stat st;
	   if (stat(date_dir.c_str(), &st) != 0) {
		   if (mkdir(date_dir.c_str(), 0755) != 0 && errno != EEXIST) {
			   std::cerr << "Failed to create directory '" << date_dir
						 << "': " << std::strerror(errno) << std::endl;
		   }
	   }
	   return date_dir;
   }

   // 实际写日志：文件路径为 {date_dir}/alarm.log
   static void WriteLog(const std::string& level,
						const std::string& rtsp_path,
						const std::string& alg_name,
						const std::string& msg)
   {
	   auto now = std::chrono::system_clock::now();
	   auto now_time_t = std::chrono::system_clock::to_time_t(now);
	   std::tm now_tm;
	   localtime_r(&now_time_t, &now_tm);

	   const std::string date_dir = GetOrMakeTodayDir(now_tm);
	   const std::string log_file = date_dir + "/alarm.log";

	   // 时间戳（到毫秒）
	   auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
					 now.time_since_epoch()) % 1000;

	   std::ostringstream line;
	   line << std::setfill('0')
			<< "[" << (now_tm.tm_year + 1900)
			<< "-" << std::setw(2) << (now_tm.tm_mon + 1)
			<< "-" << std::setw(2) << now_tm.tm_mday
			<< " " << std::setw(2) << now_tm.tm_hour
			<< ":" << std::setw(2) << now_tm.tm_min
			<< ":" << std::setw(2) << now_tm.tm_sec
			<< "." << std::setw(3) << ms.count() << "]"
			<< " [" << level << "]";

	   if (!rtsp_path.empty())
		   line << " [cam=" << rtsp_path << "]";
	   if (!alg_name.empty())
		   line << " [alg=" << alg_name << "]";

	   line << " " << msg;

	   // 线程安全写文件（每次打开append，避免长时间持有fd）
	   std::lock_guard<std::mutex> lk(log_mutex);
	   std::ofstream ofs(log_file, std::ios::app);
	   if (ofs) {
		   ofs << line.str() << "\n";	// 明确换行
		   ofs.flush(); 				 // 立即落盘
	   } else {
		   // 打不开文件，降级到stderr
		   std::cerr << line.str() << std::endl;
	   }
   }


};

#endif

