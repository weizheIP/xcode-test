

#include <vector>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <thread>
#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
// #include <pthread.h>
#include <map>
#include <ctime>
#include <memory>
// #include "object_detect.h"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include <bits/stdint-uintn.h>

#include "post/queue.h"
#include "common/http/httplib.h"
#include "common/ConvertImage.h"

#if defined(RK) || defined(X86_TS)
// #ifdef RK
#include "prefer/get_videoffmpeg.hpp" //getframe要反
#else
#include "prefer/get_video1.hpp"
#endif

// #include "common/track/SORT.h"
// #include "Track/inc/SORT.h"
#include "post/model_base.h"
#include "post/inferModel/customModel-new.h"
#include "post/face/face_db.h"

#include "post/car/car_thread.h"

#include "app_postprocess.h"
#include "license.h"
// #include "post/algsCom/cdsServer.h"
// #include "post/algsCom/httpGetImage.h"

using namespace std;

BlockingQueue<std::map<std::string, cv::Mat>> g_images(16);
BlockingQueue<std::map<std::string, cv::Mat>> g_images2(18);
BlockingQueue<std::map<std::string, cv::Mat>> g_preview_images(7);
// BlockingQueue<std::map<std::string, cv::Mat>> g_capture_images(30);

// std::map<std::string, cv::Mat> global_images;
// std::mutex global_images_mtx;
// json global_configs, global_config_preview;
// std::mutex global_configs_mtx,global_config_preview_mtx;
string ip_port = "192.168.18.60:9696";
string sTypeName = "";
int gGpuSize = 1;
int cams_count = 0; // 全局摄像头数量
float MaxConfLimit = 1.72;

std::shared_ptr<FaceServer> faceDb;

extern BlockingQueue<message_img> queue_pre;
extern std::map<std::string, std::shared_ptr<BlockingQueue<message_det>>> queue_post;
extern std::mutex queue_post_lock;


// 视频解码线程
class VideoDecoder
{
public:
  VideoDecoder(std::string _rtsp_path, int _gpuid = 0)
  {
    rtsp_path = _rtsp_path;
    gpuId = _gpuid % gGpuSize;
    camId = _gpuid;
    // 打开视频文件
    thread = std::thread(&VideoDecoder::decode, this);
  }

  ~VideoDecoder()
  {
    stop = true;
    thread.join();
  }

  void decode()
  {
    while (!stop)
    {
      try
      {
        // cv::VideoCapture capture(rtsp_path); //cpu 解码性能不够，会内存上涨
        VideoDeal capture(rtsp_path, gpuId);
        // cv::Mat frame;
        if (capture.status)
          update_camera_status(rtsp_path, 1, "");
        else
        {
          update_camera_status(rtsp_path, 0, "open cam error.");
          std::cerr << "[ERROR] cam open error: " << rtsp_path << '\n';
          sleep(2);
          continue;
        }
        std::time_t tv1 = 0, tv2 = 0, tv3 = 0, tv4 = 0, tv5 = 0, tv6 = 0;
        int FalseTime = 0;
        #ifdef X86_TS
        capture.cams_count = cams_count; // 全局摄像头数量
        #endif
        while (!stop)
        {
          tv3 = std::time(0)-1;
          if (tv3 - tv4 >= 60)
          {
            tv4 = tv3;
            cout << "[INFO] time cam decode1:  " << rtsp_path << "--" << g_images.GetSize()<<"-"<<tv4 << endl;
          }
          // 读取视频帧
          cv::Mat frame; // 引用计数，别的线程在用就不会释放，防止偏移
          // if (!capture.read(frame))
          #if defined(RK) || defined(X86_TS)
          int getRet = capture.getframe(frame);
          if (getRet)
          {
            // if (getRet == 1)
            // {        
              usleep(40*1000);        
              FalseTime++;
              if(FalseTime > 100){   
                FalseTime = 0;
                cout << "[ERROR] cam read error 60s:  " << rtsp_path << frame.size() << endl;
                throw "!";
              }                
              continue;
            // }
            // cout << "[ERROR] cam:  " << rtsp_path << frame.size() << endl;
            // throw "!";
          }
          FalseTime = 0;
          #else          
          if (!capture.getframe(frame))
          {
            cout << "[ERROR] cam:  " << rtsp_path << frame.size() << endl;
            throw "!";
          }
          #endif
          tv5 = std::time(0);
          if (tv5 - tv6 >= 60)
          {
            tv6 = tv5;
            cout << "[INFO] time cam decode2:  " << rtsp_path << "--" << g_images.GetSize()<<"-"<<tv6 << endl;
          }
          // {
          //   std::lock_guard<std::mutex> lock(global_images_mtx);
          //   global_images[rtsp_path] = frame;
          // }
          std::map<std::string, cv::Mat> aa;
          aa[rtsp_path] = frame;
          if (camId % 2 == 0)
            g_images.Push(aa);
          else
            g_images2.Push(aa);
          g_preview_images.Push(aa);
          usleep(5);
          // tv2 = std::time(0);
          // if (tv2 - tv1 >= 60)
          // {
          //   tv1 = tv2;
          //   cout << "[INFO] time cam decode:  " << rtsp_path << " - " << camId << " - " << frame.size() << "--" << g_images.GetSize() << endl;
          // }
        }
      }
      catch (...)
      {
        update_camera_status(rtsp_path, 0, "open cam error.");
        std::cerr << "[ERROR] cam throw error: " << rtsp_path << '\n';
        sleep(2);
      }
    }
  }

private:
  std::thread thread;
  std::atomic_bool stop{false};
  string rtsp_path;
  int gpuId = 0;
  int camId = 0;
};

// 业务后处理线程
class Processor
{
public:
  Processor(string _alg_name, string _rtsp_path)
  {
    alg_name = _alg_name;
    rtsp_path = _rtsp_path;
    thread = std::thread(&Processor::process, this);
  }

  ~Processor()
  {
    stop = true;
    thread.join();
  }

  void process()
  {
    while (!stop)
    {
      try
      {
        PostProcessor postprocess(alg_name, rtsp_path);
        json cam_config, _configs;
        while (!stop)
        {
          outDetectData result;
          if (inputQueue.IsEmpty())
          {
            usleep(10000);
            continue;
          }
          if (inputQueue.Pop(result) != 0)
          {
            usleep(10000);
            continue;
          }

          int cam_config_open = 0;
          {
            _configs = result.configs;
            // std::lock_guard<std::mutex> lock(global_configs_mtx);
            for (auto j_alg : _configs["camera_all"])
            {
              if (alg_name == j_alg["algorithm"])
              {
                for (auto jx : j_alg["cams"])
                {
                  if (rtsp_path == jx["rtsp"])
                  {
                    cam_config = jx;
                  }
                }
              }
            }
          }
          {
            // std::lock_guard<std::mutex> lock(global_config_preview_mtx);
            // if (rtsp_path == global_config_preview["rtsp"] and alg_name == global_config_preview["alg_id"])
            if (rtsp_path == _configs["ws_push"]["rtsp"] and alg_name == _configs["ws_push"]["alg_id"])
              cam_config_open = 1;
          }
          usleep(1000);
          // outDetectData result;
          // if (inputQueue.IsEmpty())
          // {
          //   usleep(20000);
          //   continue;
          // }
          // if (inputQueue.Pop(result) != 0)
          // {
          //   usleep(10000);
          //   continue;
          // }
          // sleep(3);

          auto start = std::chrono::steady_clock::now();
          if (alg_name == "student_absent_detection" || alg_name == "leaving_post" || alg_name == "sleeping_post"  || alg_name == "cabinet_indicator_light_detection" || alg_name == "cv_asset_deficiency_detection" || alg_name == "cv_sales_terminal_out_location")
            postprocess.GatheringProcess(result, cam_config, cam_config_open);
          else if (alg_name == "people_gathering_detection" || alg_name == "traffic_jam_detection" || alg_name == "cv_double_banknote_detection" || alg_name == "cv_site_personnel_limit")
            postprocess.GatheringProcess(result, cam_config, cam_config_open);
          else if (alg_name == "loitering_detection" || alg_name == "cv_wandering")
            postprocess.IdProcess(result, cam_config, cam_config_open);
          else if (alg_name == "illegal_parking_detection")
            postprocess.IllegalParkingProcess(result, cam_config, cam_config_open);
          else if (alg_name == "empty_parking_space_detection")
            postprocess.EmptyParkingSpace(result, cam_config, cam_config_open);
          else if (alg_name == "pointer_instrument_detection")
            postprocess.MechanicalProcess(result, cam_config, cam_config_open);
          else if (alg_name == "cv_illegal_parking_across_parking_spaces")
            postprocess.AcrossPakingProces(result, cam_config, cam_config_open);
          // else if (alg_name == "face_detection")
          else if (alg_name == "face_detection" || alg_name == "personnel_duty_detection" || alg_name == "anti_theft_detection" || alg_name == "stranger_use_terminal_detection" \
                  || alg_name == "face_tracking")
            postprocess.FaceProcess(result, cam_config, cam_config_open);
		  else if (alg_name == "license_plate_tracking" || alg_name == "license_plate_detection")
			  postprocess.LicensePlateProcess(result, cam_config, cam_config_open);
          else
            postprocess.NoIdProcess(result, cam_config, cam_config_open);
          auto end = std::chrono::steady_clock::now();
          std::chrono::duration<double> elapsed_seconds = end - start;
          if (cam_config_open == 1)
            cout << "[FPS] time show:  " << alg_name << " " << rtsp_path << " " << elapsed_seconds.count() << endl;
        }
      }
      catch (...)
      {
        std::cerr << "[ERROR] post error: " << alg_name << " " << rtsp_path << '\n';
        sleep(1);
      }
    }
  }

  BlockingQueue<outDetectData> inputQueue;

private:
  std::thread thread;
  std::atomic_bool stop{false};
  string rtsp_path, alg_name;
};

class PreviewThread
{
public:
  PreviewThread(string _rtsp_path)
  {
    rtsp_path = _rtsp_path;
    thread = std::thread(&PreviewThread::process, this);
  }

  ~PreviewThread()
  {
    stop = true;
    thread.join();
  }

  void process()
  {
    while (!stop)
    {
      try
      {
        PostProcessor postprocess("", rtsp_path);
        while (!stop)
        {
          cv::Mat frame;
          // {
          //   std::lock_guard<std::mutex> lock(global_images_mtx);
          //   frame = global_images[rtsp_path];
          // }
          std::map<std::string, cv::Mat> _rtsp_image;
          if (g_preview_images.IsEmpty())
          {
            usleep(20000);
            continue;
          }
          if (g_preview_images.Pop(_rtsp_image) != 0)
          {
            usleep(10000);
            continue;
          }
          frame = _rtsp_image[rtsp_path];

          if (frame.empty())
            continue;
#ifdef DECNOTRESIZE
          sw_resize(frame);
#endif 
          postprocess.Preview(frame);
          usleep(10*1000);
          cout << "[FPS] preview: " << rtsp_path << endl;
        }
      }
      // catch (const std::exception &e)
      catch (...)
      {
        std::cerr << "[ERROR] preview error " << rtsp_path << '\n';
        sleep(1);
      }
    }
  }

private:
  std::thread thread;
  std::atomic_bool stop{false};
  string rtsp_path;
};

// 算法线程类
class AlgorithmThread
{
public:
  AlgorithmThread(string _algName, int _gpuid = 0)
  {
    algName = _algName;
    gpuId = _gpuid % gGpuSize;
    thread = std::thread(&AlgorithmThread::run, this);
    // thread = std::thread(&AlgorithmThread::run, this); //人脸可以加一个
  }

  ~AlgorithmThread()
  {
    stop = true;
    thread.join();
  }

  void run()
  {
    // try
    // {
    json _configs;
    vector<string> rtsp_urls_pre;
    // std::map<std::string, cv::Mat> rtsp_images;
    std::map<std::string, std::shared_ptr<Processor>> processors;
    detYOLO yolo_model;
    serialYOLO yolo_model_smoking;
    parallelYOLO yolo_model_garbage;
    chargeGunYOLO yolo_model_gun;
    // lmYOLO lm_model;
    
	CarInfer& carRun = CarInfer::instance();
    faceRec& facetool = faceRec::instance(); // 人脸rk可以创建2个线程用2个核心加速
    if (algName == "smoking_detection" ||  algName == "charging_pile_occupied_by_oil_car")
    {
      yolo_model_smoking.init(algName, 1, gpuId);
    }
    else if (algName == "garbage_exposure" || algName == "cv_large_item_inspection"  || algName == "phone_detection" || algName == "play_phone_detection" || algName == "illegal_operation_phone_terminal_detection")
    {
      yolo_model_garbage.init(algName, 1, gpuId);
    }
    else if (algName == "cv_illegal_parking_across_parking_spaces")
    {
      // lm_model.init(algName + "/m_enc.pt", 1, gpuId);
      yolo_model.init(algName + "/m_enc.pt", 1, gpuId);
    }
    else if (algName == "cv_charging_gun_notputback")
    {
      yolo_model_gun.init(algName, 1, gpuId);
    }

    else if (algName == "face_detection" || algName == "personnel_duty_detection" || algName == "anti_theft_detection" || algName == "stranger_use_terminal_detection" || algName == "face_tracking")
    {
      facetool.init("");
    }
	else if (algName == "license_plate_tracking" || algName == "license_plate_detection") {
	    carRun.init("", algName);
	}
    else
    {
      yolo_model.init(algName + "/m_enc.pt", 1, gpuId);
    }
    // while (!stop)
    std::time_t tv1 = 0, tv2 = 0;
    while (1)
    {
      vector<string> rtsp_urls;
      // 3. 根据绑定，把每个算法的相机进行添加和删除；后处理线程进行添加和删除
      // {
      //   std::lock_guard<std::mutex> lock(global_images_mtx);
      //   rtsp_images = global_images;
      // }

      {
        if (!inputQueue.IsEmpty())
        {
          // inputQueue.Pop(_configs, 500);
          inputQueue.Pop(_configs);
        }
        // cout<<algName<< _configs<<endl;
        // std::lock_guard<std::mutex> lock(global_configs_mtx);
        for (auto j_alg : _configs["camera_all"])
        {
          if (algName == j_alg["algorithm"])
          {
            for (auto j : j_alg["cams"])
            {
              rtsp_urls.push_back(j["rtsp"]);
            }
          }
        }
      }

      tv2 = std::time(0);
      if (tv2 - tv1 >= 60)
      {
        tv1 = tv2;
        cout << "[INFO] now alg:  " << algName << "--" << rtsp_urls.size() << endl;
      }

      // cout <<algName<< "----:"<<g_images.GetSize() << endl;
      std::map<std::string, cv::Mat> _rtsp_image5;
      for (int i = 0; i < rtsp_urls.size() * 3; i++)
      {
        std::map<std::string, cv::Mat> _rtsp_image;
        if (!g_images.IsEmpty())
        {
          if (g_images.Pop(_rtsp_image) != 0)
          {
            continue;
          }
          for (const auto &pair : _rtsp_image)
          {
            if (std::count(rtsp_urls.begin(), rtsp_urls.end(), pair.first) > 0)
              _rtsp_image5.emplace(pair);
          }
        }
        else
        {
          usleep(1000);
        }

        std::map<std::string, cv::Mat> _rtsp_image2;
        if (!g_images2.IsEmpty())
        {
          if (g_images2.Pop(_rtsp_image2) != 0)
          {
          }
          for (const auto &pair : _rtsp_image2)
          {
            if (std::count(rtsp_urls.begin(), rtsp_urls.end(), pair.first) > 0)
              _rtsp_image5.emplace(pair);
          }
        }

        std::vector<string> keys;
        for (const auto &pair : _rtsp_image5)
        {
          keys.push_back(pair.first);
        }
        std::sort(keys.begin(), keys.end());
        std::sort(rtsp_urls.begin(), rtsp_urls.end());
        if (keys == rtsp_urls)
          break;
      }

      if (rtsp_urls_pre != rtsp_urls)
      {
        // delete
        for (int i = 0; i < rtsp_urls_pre.size(); ++i)
        {
          if (count(rtsp_urls.begin(), rtsp_urls.end(), rtsp_urls_pre[i]) == 0)
          {
            processors[rtsp_urls_pre[i]].reset();
            processors.erase(rtsp_urls_pre[i]);
            cout << "[INFO] post delete: " << rtsp_urls_pre[i] << endl;
          }
        }
        // add
        for (int i = 0; i < rtsp_urls.size(); ++i)
        {
          if (count(rtsp_urls_pre.begin(), rtsp_urls_pre.end(), rtsp_urls[i]) == 0)
          {
            processors[rtsp_urls[i]] = std::make_shared<Processor>(algName, rtsp_urls[i]);
            cout << "[INFO] post add: " << rtsp_urls[i] << endl;
          }
        }
      }
      rtsp_urls_pre = rtsp_urls;
      if (stop and (rtsp_urls.size() == 0))
        break;

      // 从解码线程获取图片并进行推理+后处理
      auto start = std::chrono::steady_clock::now();
      for (const auto &pair : _rtsp_image5)
      {
        string rtsp = pair.first;
        cv::Mat image = pair.second;
        if (image.empty()){
		  std::cout << "[INFO] _rtsp_image5 image is empty!!!" << " " << algName << " " << " on: " << _rtsp_image5.size() << endl;
          continue;
        }
        // if (algName == "face_detection")
        if (algName == "face_detection" || algName == "personnel_duty_detection" || algName == "anti_theft_detection" || algName == "stranger_use_terminal_detection" || algName == "face_tracking")
        {
          json cam_config;
          for (auto j_alg : _configs["camera_all"])
          {
            if (algName == j_alg["algorithm"])
            {
              for (auto jx : j_alg["cams"])
              {
                if (rtsp == jx["rtsp"])
                {
                  cam_config = jx;
                  break;
                }
              }
              break;
            }
          }
          // vector<faceInfo> ans;
          // ans = facetool.infer(image,cam_config);
//		  vector<vector<int>> Roi = cam_config["roi"];

		  vector<vector<int>> Roi = (cam_config["roi"].size() == 2) ? cam_config["roi"][0] : cam_config["roi"];
		  
          float score = cam_config["config"]["score"];
          int limarea = cam_config["config"]["area"];

          std::vector<cv::Point> contours;
          int previewH = 1280.f / (float)image.cols * (float)image.rows;
          if (previewH % 2 != 0)
            previewH++;
          for (int i = 0; i < Roi.size(); i++)
          {
            int x = (float)Roi[i][0] * ((float)image.cols / (float)1280.f);
            int y = (float)Roi[i][1] * ((float)image.rows / (float)previewH);
            contours.push_back(cv::Point(x, y));
          }

          vector<faceInfo> ans;
          std::vector<Yolov5Face_BoxStruct> face_result;
//		  T_CurResult face_result;
          std::vector<int> delNums;
          std::vector<float> input;
          int querySize;

//		  face_result.rtsp_size = rtsp_urls.size();

	      if (algName == "face_tracking"){
//	          facetool.infer(image, contours, face_result, ans, score, limarea, rtsp);
			  facetool.infer(image, contours, face_result, ans, score, limarea);
	      }else{
//	          facetool.infer(image, contours, face_result, delNums, input, querySize, score, limarea, rtsp);
//	          faceDb->search(ans, face_result.result, delNums, input, querySize, 2);
			  facetool.infer(image, contours, face_result, delNums, input, querySize, score, limarea);
			  faceDb->search(ans, face_result, delNums, input, querySize, 2);
	      }

          // static bool has_executed = false;
          // if (!has_executed) {
          //   facetool.infer(image, contours, face_result, delNums, input, querySize, score, limarea);
          //   faceDb->search(ans, face_result, delNums, input, querySize, 2);
          //   has_executed = true;
          // }

//          outDetectData result = {face_result.img, {}, {}, ans, _configs};
          outDetectData result = {image, {}, {}, ans, _configs};
          processors[rtsp]->inputQueue.Push(result);
        }
		else if (algName == "license_plate_tracking" || algName == "license_plate_detection") {
			
			json cam_config;
			for (auto j_alg : _configs["camera_all"])
			{
			  if (algName == j_alg["algorithm"])
			  {
				for (auto jx : j_alg["cams"])
				{
				  if (rtsp == jx["rtsp"])
				  {
					cam_config = jx;
					break;
				  }
				}
				break;
			  }
			}
			
			vector<vector<int>> Roi = cam_config["roi"];
			float score = cam_config["config"]["score"];
//			int limarea = cam_config["config"]["area"];

            vector<BoxInfo> carBoxInfoVec;
            carBoxInfoVec = carRun.carInfer(image);

			std::vector<cv::Point> contours;
			int previewH = 1280.f / (float)image.cols * (float)image.rows;
			if (previewH % 2 != 0)
			  previewH++;
			for (int i = 0; i < Roi.size(); i++)
			{
			  int x = (float)Roi[i][0] * ((float)image.cols / (float)1280.f);
			  int y = (float)Roi[i][1] * ((float)image.rows / (float)previewH);
			  contours.push_back(cv::Point(x, y));
			}
			RoiFillter_car(carBoxInfoVec, contours);
			
			//LOG_DEBUG("");
			vector<BoxInfo> generate_boxes;
			for (auto x : carBoxInfoVec)
			{
				if (score > x.score)
					continue;
				if ((algName == "license_plate_tracking" || algName == "license_plate_detection") && x.label != 0)
					continue;
				generate_boxes.push_back(x);
			}
			outDetectData result = {image, generate_boxes, {}, {}, _configs};
			processors[rtsp]->inputQueue.Push(result);
		}
        else
        {
          #ifdef DECNOTRESIZE
          sw_resize(image);
          #endif 
          std::vector<BoxInfo> boxes;
          if (algName == "smoking_detection" || algName == "charging_pile_occupied_by_oil_car")
            boxes = yolo_model_smoking.infer(image);
          else if (algName == "garbage_exposure" || algName == "cv_large_item_inspection"  || algName == "phone_detection" || algName == "play_phone_detection" || algName == "illegal_operation_phone_terminal_detection")
            boxes = yolo_model_garbage.infer(image);
          else if (algName == "cv_charging_gun_notputback")
          {
            json cam_config;
            for (auto j_alg : _configs["camera_all"])
            {
              if (algName == j_alg["algorithm"])
              {
                for (auto jx : j_alg["cams"])
                {
                  if (rtsp == jx["rtsp"])
                  {
                    cam_config = jx;
                    break;
                  }
                }
                break;
              }
            }
            boxes = yolo_model_gun.infer(image, cam_config);
          }
          else
            boxes = yolo_model.infer(image);
          outDetectData result = {image, boxes, {}, {}, _configs};
          processors[rtsp]->inputQueue.Push(result);
        }
        // auto end2 = std::chrono::steady_clock::now();
        // std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
        // cout << "[FPS] time infer one: " << " " << algName << " " << elapsed_seconds2.count() << endl; //相机多推理就很慢
      }
      auto end = std::chrono::steady_clock::now();
      std::chrono::duration<double> elapsed_seconds = end - start;
      if (elapsed_seconds.count() > 0.005)
        cout << "[FPS] time infer: " << " " << algName << " " << elapsed_seconds.count() << " fps: " << 1 / elapsed_seconds.count() << " on: " << _rtsp_image5.size() << endl;
      // if (elapsed_seconds.count() < 0.005)
      //   sleep(0.1);
    }
    // }
    // catch(...)
    // {
    //   std::cerr << "[ERROR] alg error: "  << '\n';
    // }
  }

  BlockingQueue<json> inputQueue;

private:
  std::thread thread;
  std::atomic_bool stop{false};
  string algName;
  int gpuId = 0;
};

class Pipeline
{
public:
  void run()
  {
    vector<string> alg_all_pre, rtsp_urls_pre, preview_urls_pre;
    std::map<std::string, std::shared_ptr<VideoDecoder>> Cams_APP_list; // 相机map<rtsp,thread>
    std::map<std::string, std::shared_ptr<AlgorithmThread>> Algs_APP_list; // 相机map<算法,thread>
    std::map<std::string, std::shared_ptr<PreviewThread>> Preview_list;

    // json _configs_pre,_wsPushConfig;
    json configs_pre;

    vector<string> alg_filter = {"crowd_density_statistics", "cv_ship_count_statistics", "cv_personnel_flow_statistics", "pointer_instrument_detection2",
      "tdds", "cv_fish_tdds", "cv_tdds_piggy", "tdgz", "dbds", "dbgz", "cv_count_chicken", "cv_tdgz", "cv_fence_dbgz","cv_fence_dbds",
      "cv_count_chicks", "chicken_coop_count", "cv_count_egg", "cattle", "cattle_ds", "cattle_tdgz", "cv_dead_amoy_detection", 
      "cv_traffic_flow", "cv_large_car_detection", "cv_parking_license_plate_detection", "cv_road_parking_license_plate_detection", "cv_vehicle_departure_detection", "cv_livestock_license_plates_detection"};
    // int cds1_status = 0;
    // std::shared_ptr<CDS_SERVER> cds1;
    // int cds_boat_status = 0;
    // std::shared_ptr<CDS_SERVER> cds_boat;
    // HttpVideoGetImage getImg;
    // faceDb = std::make_shared<FaceServer>("../../civiapp/faceDatabase");
    faceDb = std::make_shared<FaceServer>("../civiapp/faceDatabase");

#ifdef X86_TS
    size_t numDevices = torch::cuda::device_count();
    gGpuSize = numDevices;

    // int deviceCount = 0;
    // cudaGetDeviceCount(&deviceCount);
    // gGpuSize = deviceCount;
    // std::cout << "[INFO] find " << gGpuSize << " device can use !" << std::endl;
#endif
    std::time_t tv1 = 0, tv2 = 0;
    while (1)
    {
      try
      {
        json configs;
        tv2 = std::time(0);
        if (tv2 - tv1 >= 60)
        {
          tv1 = tv2;
          cout << "[INFO] time main -------------------------------------------------------- " << endl;
        }
        vector<string> alg_all, rtsp_urls, preview_urls;
        std::set<string> rtsp_urls_set;
        // 1. http接口
        if (auto res = httplib::Client(ip_port.c_str()).Post("/api/camera/all"))
        {
          auto data = res->body;
          json j = json::parse(data)["data"];
          configs["camera_all"] = j;
          // std::lock_guard<std::mutex> lock(global_configs_mtx);
          // global_configs = j;
          for (auto it : j)
          {
            if (std::find(alg_filter.begin(), alg_filter.end(), it["algorithm"]) == alg_filter.end())
            {
              if (sTypeName != "")
              {
                if (it["algorithm"] == sTypeName)
                  alg_all.push_back(it["algorithm"]);
              }
              else
                alg_all.push_back(it["algorithm"]);
            }
            // if (it["algorithm"] == "crowd_density_statistics")
            // {
            //   if (cds1_status == 0)
            //   {
            //     cds1 = std::make_shared<CDS_SERVER>("crowd_density_statistics", 58081);
            //     cds1_status = 1;
            //     sleep(1);
            //   }
            // }
            // if (it["algorithm"] == "cv_ship_count_statistics")
            // {
            //   if (cds_boat_status == 0)
            //   {
            //     cds_boat = std::make_shared<CDS_SERVER>("cv_ship_count_statistics", 58084);
            //     cds_boat_status = 1;
            //     sleep(1);
            //   }
            // }
          }
          for (const auto &item : j)
          {
            const auto &cams = item.value("cams", json::array());
            for (const auto &cam : cams)
            {
              const std::string rtsp = cam.value("rtsp", "");
              if (!rtsp.empty())
              {
                rtsp_urls_set.insert(rtsp);
              }
            }
          }
        }
        else
        {
          cout << "[Error] http1: " << res.error() << endl;
          continue;
        }
        if (auto res = httplib::Client(ip_port.c_str()).Get("/api/ws/getPushVideo"))
        {
          auto data = res->body;
          json j = json::parse(data)["data"];
          configs["ws_push"] = j;
          // std::lock_guard<std::mutex> lock(global_config_preview_mtx);
          // global_config_preview = j;
        }
        else
        {
          cout << "[Error] http2: " << res.error() << endl;
          continue;
        }
        if (auto res = httplib::Client(ip_port.c_str()).Get("/api/camera/preview/rtsp"))
        {
          auto data = res->body;
          json j = json::parse(data)["data"];
          for (auto j_rtsp : j)
          {
            rtsp_urls_set.insert(j_rtsp["rtsp"]);
            preview_urls.push_back(j_rtsp["rtsp"]);
          }
        }
        else
        {
          cout << "[Error] http3: " << res.error() << endl;
          continue;
        }
        for (const auto &value : rtsp_urls_set)
        {
          rtsp_urls.push_back(value);
        }
        cams_count = rtsp_urls.size();

        // 2.相机增删
        if (rtsp_urls_pre != rtsp_urls)
        {
          // delete
          for (int i = 0; i < rtsp_urls_pre.size(); ++i)
          {
            if (count(rtsp_urls.begin(), rtsp_urls.end(), rtsp_urls_pre[i]) == 0)
            {
              Cams_APP_list[rtsp_urls_pre[i]].reset();
              Cams_APP_list.erase(rtsp_urls_pre[i]);
              cout << "[INFO] cam delete: " << rtsp_urls_pre[i] << endl;
              sleep(1);
			  DeleteCarPost(rtsp_urls_pre[i]);		// 等待车牌识别处理完当前任务
            }
          }
          // add
          for (int i = 0; i < rtsp_urls.size(); ++i)
          {
            if (count(rtsp_urls_pre.begin(), rtsp_urls_pre.end(), rtsp_urls[i]) == 0)
            {
			  MallocCarPost(rtsp_urls[i]);
              Cams_APP_list[rtsp_urls[i]] = std::make_shared<VideoDecoder>(rtsp_urls[i], i);
              cout << "[INFO] cam add: " << rtsp_urls[i] << "----: " << i << "----: " << (i % gGpuSize) << endl;
              sleep(1);
            }
          }
        }
        // else
        // {
        //   sleep(1);
        // }
        rtsp_urls_pre = rtsp_urls;
        
        // 3. 算法增删
        if (alg_all_pre != alg_all)
        {
          // delete
          for (int i = 0; i < alg_all_pre.size(); ++i)
          {
            if (count(alg_all.begin(), alg_all.end(), alg_all_pre[i]) == 0)
            {
              if (configs_pre != configs)
              {
                for (auto &[key, value] : Algs_APP_list)
                {
                  value->inputQueue.Push(configs);
                }
              }
              Algs_APP_list[alg_all_pre[i]].reset();
              Algs_APP_list.erase(alg_all_pre[i]);
              cout << "[INFO] alg delete: " << alg_all_pre[i] << endl;
              sleep(1);
            }
          }
          // add
          for (int i = 0; i < alg_all.size(); ++i)
          {
            if (count(alg_all_pre.begin(), alg_all_pre.end(), alg_all[i]) == 0)
            {
              Algs_APP_list[alg_all[i]] = std::make_shared<AlgorithmThread>(alg_all[i], i);
              cout << "[INFO] alg add: " << alg_all[i] << "----: " << i << "----: " << (i % gGpuSize) << endl;
              sleep(1);
            }
          }
        }
        // else
        // {
        //   sleep(1);
        // }
        // sleep(5);
        if (configs_pre != configs)
        {
          for (auto &[key, value] : Algs_APP_list)
          {
            value->inputQueue.Push(configs);
          }
        }
        alg_all_pre = alg_all;
        configs_pre = configs;

        // 4.preview增删
        if (preview_urls_pre != preview_urls)
        {
          // delete
          for (int i = 0; i < preview_urls_pre.size(); ++i)
          {
            if (count(preview_urls.begin(), preview_urls.end(), preview_urls_pre[i]) == 0)
            {
              Preview_list[preview_urls_pre[i]].reset();
              Preview_list.erase(preview_urls_pre[i]);
              cout << "[INFO] preview delete: " << preview_urls_pre[i] << endl;
            }
          }
          // add
          for (int i = 0; i < preview_urls.size(); ++i)
          {
            if (count(preview_urls_pre.begin(), preview_urls_pre.end(), preview_urls[i]) == 0)
            {
              Preview_list[preview_urls[i]] = std::make_shared<PreviewThread>(preview_urls[i]);
              cout << "[INFO] preview add: " << preview_urls[i] << endl;
            }
          }
        }
        // else
        // {
        //   sleep(1);
        // }
        usleep(50*1000);
        preview_urls_pre = preview_urls;
      }
      catch (...)
      {
        std::cerr << "[ERROR] init error " << '\n';
        preview_urls_pre = vector<string>{};
        rtsp_urls_pre = vector<string>{};
        alg_all_pre = vector<string>{};
        sleep(2);
      }
    }
  }
};

int main(int argc, char *argv[])
{
#ifndef DEED_DEBUG
  if (!checklic("../civiapp/licens/license.lic"))
  {
    std::cout << "[Error] Check license failed!" << std::endl;
    return 0;
  }
#endif
  cout << "\n Usage : " << argv[0] << " [alarm_video : 0|1] [ip_port : 127.0.0.1:9696]  [one_sTypeName : face_detection]" << endl;
  ip_port = "127.0.0.1:9696";
  if (argc >= 2){
  	  AlarmVideoInfo::switchAlarm = std::stoi(argv[1]);
  }

  if (argc == 3)
  {
    ip_port = argv[2];
  }
  if (argc == 4)
  {
    ip_port = argv[2];
    sTypeName = argv[3];
  }
  cout << "[INFO] start_app --------------------------------------------------------*** " << endl;
  Pipeline pipeline;
  pipeline.run();
  return 1;
}


