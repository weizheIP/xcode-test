// 人脸检测，有进行人脸识别 （不用跟踪）

#pragma once

#include <mutex>

#include "opencv2/core.hpp"
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <iostream>
#include <vector>
#include "post/base.h"
// #include "post/post_util.h"
// #include "post/algsCom/post_util.h"
#include "post/model_base.h"
#include "post/queue.h"
using namespace cv;

using idx_t = faiss::idx_t;

struct faceInfo
{
    float x;
    float y;
    float w;
    float h;
    float conf;
    float score;
    std::vector<cv::Point> keypoint;
    std::string fileName;
    size_t id;
    idx_t spId;
	std::vector<float> feature;
};
// 1：闭眼 0：睁眼 rk
static bool EyeCloseStatus(std::vector<cv::Point> landmarks)
{

    float EAR_left = (sqrt(pow(fabs(landmarks[41].x - landmarks[36].x), 2) + pow(fabs(landmarks[41].y - landmarks[36].y), 2)) + sqrt(pow(fabs(landmarks[42].x - landmarks[37].x), 2) + pow(fabs(landmarks[42].y - landmarks[37].y), 2))) / (2 * sqrt(pow(fabs(landmarks[39].x - landmarks[35].x), 2) + pow(fabs(landmarks[39].y - landmarks[35].y), 2)));

    float EAR_right = (sqrt(pow(fabs(landmarks[95].x - landmarks[90].x), 2) + pow(fabs(landmarks[95].y - landmarks[90].y), 2)) + sqrt(pow(fabs(landmarks[96].x - landmarks[91].x), 2) + pow(fabs(landmarks[96].y - landmarks[91].y), 2))) / (2 * sqrt(pow(fabs(landmarks[93].x - landmarks[89].x), 2) + pow(fabs(landmarks[93].y - landmarks[89].y), 2)));
    // cv::imwrite("result1.jpg", orig_img);

    // if (EAR_left<0.2 && EAR_right<0.2) {
    if ((EAR_left + EAR_right) / 2 < 0.18)
    {
        std::cout << "CLOSE EAR_left: " << (EAR_left + EAR_right) / 2 << "," << EAR_left << "    EAR_right: " << EAR_right << std::endl;
        //       std::cout << "闭眼" << std::endl;
        return 1;
    }
    else
    {
        std::cout << "Open EAR_left: " << (EAR_left + EAR_right) / 2 << "," << EAR_left << "    EAR_right: " << EAR_right << std::endl;
        return 0;
    }
}

static int face_quality(cv::Mat frame, Yolov5Face_BoxStruct generate_boxes, int limarea)
{
    cv::Mat im = frame.clone();
//    cv::Mat im = frame;
    std::vector<cv::Point2d> image_points;
    image_points.push_back(generate_boxes.keypoint[0]); // Left eye left corner
    image_points.push_back(generate_boxes.keypoint[1]); // Right eye right corner
    image_points.push_back(generate_boxes.keypoint[2]); // Nose tip
    image_points.push_back(generate_boxes.keypoint[3]); // Left Mouth corner
    image_points.push_back(generate_boxes.keypoint[4]); // Right mouth corner

    // 3D model points.
    std::vector<cv::Point3d> model_points;
    model_points.push_back(cv::Point3d(-165.0f, 170.0f, -115.0f));  // Left eye left corner
    model_points.push_back(cv::Point3d(165.0f, 170.0f, -115.0f));   // Right eye right corner
    model_points.push_back(cv::Point3d(0.0f, 0.0f, 0.0f));          // Nose tip
    model_points.push_back(cv::Point3d(-150.0f, -150.0f, -125.0f)); // Left Mouth corner
    model_points.push_back(cv::Point3d(150.0f, -150.0f, -125.0f));  // Right mouth corner

    // Camera internals
    double focal_length = im.cols; // Approximate focal length.
    cv::Point2d center = cv::Point2d(im.cols / 2, im.rows / 2);
    cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal_length, 0, center.x, 0, focal_length, center.y, 0, 0, 1);
    cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type); // Assuming no lens distortion

    // Output rotation and translation
    cv::Mat rotation_vector; // Rotation in axis-angle form
    cv::Mat translation_vector;
    // Solve for pose
    cv::solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector, false, 1); // opencv4; if 3,->0

    cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1); // 3 x 4 R | T
    cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);
    cv::Mat rotation_matrix;
    cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_translation = cv::Mat(4, 1, CV_64FC1);
    cv::Rodrigues(rotation_vector, rotation_matrix);
    cv::hconcat(rotation_matrix, translation_vector, pose_mat);
    cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);
    // cout <<euler_angle<< endl;
    double pitch = euler_angle.at<double>(0); // 俯仰角
    double yaw = euler_angle.at<double>(1);	  // 转头角
    double roll = euler_angle.at<double>(2);  // 歪头角度

    if (pitch > 0)
    {
        pitch = 180 - pitch;
    }
    else
    {
        pitch = -(180 + pitch);
    }
    // cout << "pitch: "<<pitch<< " yaw: " << yaw <<  " roll: "<<  roll << endl;

    //  # 清晰度
    cv::Mat gray;
    cv::cvtColor(im, gray, cv::COLOR_BGR2GRAY);
    cv::Rect roiRect((int)generate_boxes.x1, (int)generate_boxes.y1, int(generate_boxes.x2 - generate_boxes.x1), int(generate_boxes.y2 - generate_boxes.y1));
    cv::Mat roi = gray(roiRect);
    int w = generate_boxes.x2 - generate_boxes.x1;
    int h = generate_boxes.y2 - generate_boxes.y1;
    if (roi.empty()) //?rk为啥空
        return 0;
    if (w > 112 || h > 112)
    {
        cv::resize(roi, roi, cv::Size(112, 112), cv::INTER_AREA);
    }
    cv::Mat laplacian;
    cv::Laplacian(roi, laplacian, CV_64F);
    cv::Scalar mean, stdDev;
    cv::meanStdDev(laplacian, mean, stdDev);
    double variance = stdDev.val[0] * stdDev.val[0];
    // cout << "清晰度: "<< variance << endl;

    // bool noseInMiddle = (generate_boxes.kpt3.pt.x > generate_boxes.kpt1.pt.x && generate_boxes.kpt2.pt.x > generate_boxes.kpt3.pt.x)&&(generate_boxes.kpt3.pt.x > generate_boxes.kpt4.pt.x && generate_boxes.kpt5.pt.x > generate_boxes.kpt3.pt.x);
    bool noseInMiddle = (generate_boxes.keypoint[2].x > generate_boxes.keypoint[0].x && generate_boxes.keypoint[1].x > generate_boxes.keypoint[2].x) && (generate_boxes.keypoint[2].x > generate_boxes.keypoint[3].x && generate_boxes.keypoint[4].x > generate_boxes.keypoint[2].x);
//	// 要显示的文字
//	std::vector<std::string> infos;
//	infos.push_back("noseInMiddle: " + std::string(noseInMiddle ? "true" : "false"));
//	infos.push_back("pitch: " + std::to_string(pitch));
//	infos.push_back("yaw: " + std::to_string(yaw));
//	infos.push_back("roll: " + std::to_string(roll));
//	infos.push_back("variance: " + std::to_string(variance));
//	infos.push_back("w: " + std::to_string(w));
//	infos.push_back("h: " + std::to_string(h));
//	infos.push_back("ratio: " + std::to_string((float)w / (float)h));
//	infos.push_back("limarea: " + std::to_string(limarea));
//	
//	// 在左上角逐行绘制
//	int baseLine = 20;
//	for (size_t i = 0; i < infos.size(); i++) {
//		cv::putText(im, infos[i], cv::Point(generate_boxes.x1 + w + 10, generate_boxes.y1 + baseLine + i * 20),
//					cv::FONT_HERSHEY_SIMPLEX,	// 字体
//					0.5,						// 字体大小
//					cv::Scalar(0, 0, 255),		// 字体颜色 (绿色)
//					1,							// 线条厚度
//					cv::LINE_AA);				// 抗锯齿
//	}
    // 人脸方向、清晰度、大小满足要求才行 variance=160 rk的yaw会到46
     if (noseInMiddle && -50 < pitch and pitch < 50 and -50 < yaw and yaw < 50 and -20 < roll and roll < 20 and variance > 100 and w > limarea and h > limarea and ((float)w / (float)h) > 0.60)
     {
         return 1;
     }
     else
     {
         // std::cout<<"noseInMiddle:"<<noseInMiddle<<",w:"<<w<<",h:"<<h<<",w/h:"<<((float)w / (float)h)<<",ptich:"<<pitch<<","<<"yaw:"<<yaw<<","<<"roll:"<<roll<<","<<"variance:"<<variance<<"\n";
         return 0;
     }
};

static cv::Mat getSimilarityTransformMatrix(float src[5][2])
{
    float dst[5][2] = {{38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f}, {41.5493f, 92.3655f}, {70.7299f, 92.2041f}};
    float avg0 = (src[0][0] + src[1][0] + src[2][0] + src[3][0] + src[4][0]) / 5;
    float avg1 = (src[0][1] + src[1][1] + src[2][1] + src[3][1] + src[4][1]) / 5;
    // Compute mean of src and dst.
    float src_mean[2] = {avg0, avg1};
    float dst_mean[2] = {56.0262f, 71.9008f};
    // Subtract mean from src and dst.
    float src_demean[5][2];
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            src_demean[j][i] = src[j][i] - src_mean[i];
        }
    }
    float dst_demean[5][2];
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            dst_demean[j][i] = dst[j][i] - dst_mean[i];
        }
    }
    double A00 = 0.0, A01 = 0.0, A10 = 0.0, A11 = 0.0;
    for (int i = 0; i < 5; i++)
        A00 += dst_demean[i][0] * src_demean[i][0];
    A00 = A00 / 5;
    for (int i = 0; i < 5; i++)
        A01 += dst_demean[i][0] * src_demean[i][1];
    A01 = A01 / 5;
    for (int i = 0; i < 5; i++)
        A10 += dst_demean[i][1] * src_demean[i][0];
    A10 = A10 / 5;
    for (int i = 0; i < 5; i++)
        A11 += dst_demean[i][1] * src_demean[i][1];
    A11 = A11 / 5;
    Mat A = (Mat_<double>(2, 2) << A00, A01, A10, A11);
    double d[2] = {1.0, 1.0};
    double detA = A00 * A11 - A01 * A10;
    if (detA < 0)
        d[1] = -1;
    double T[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    Mat s, u, vt, v;
    SVD::compute(A, s, u, vt);
    double smax = s.ptr<double>(0)[0] > s.ptr<double>(1)[0] ? s.ptr<double>(0)[0] : s.ptr<double>(1)[0];
    double tol = smax * 2 * FLT_MIN;
    int rank = 0;
    if (s.ptr<double>(0)[0] > tol)
        rank += 1;
    if (s.ptr<double>(1)[0] > tol)
        rank += 1;
    double arr_u[2][2] = {{u.ptr<double>(0)[0], u.ptr<double>(0)[1]}, {u.ptr<double>(1)[0], u.ptr<double>(1)[1]}};
    double arr_vt[2][2] = {{vt.ptr<double>(0)[0], vt.ptr<double>(0)[1]}, {vt.ptr<double>(1)[0], vt.ptr<double>(1)[1]}};
    double det_u = arr_u[0][0] * arr_u[1][1] - arr_u[0][1] * arr_u[1][0];
    double det_vt = arr_vt[0][0] * arr_vt[1][1] - arr_vt[0][1] * arr_vt[1][0];
    if (rank == 1)
    {
        if ((det_u * det_vt) > 0)
        {
            Mat uvt = u * vt;
            T[0][0] = uvt.ptr<double>(0)[0];
            T[0][1] = uvt.ptr<double>(0)[1];
            T[1][0] = uvt.ptr<double>(1)[0];
            T[1][1] = uvt.ptr<double>(1)[1];
        }
        else
        {
            double temp = d[1];
            d[1] = -1;
            Mat D = (Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
            Mat Dvt = D * vt;
            Mat uDvt = u * Dvt;
            T[0][0] = uDvt.ptr<double>(0)[0];
            T[0][1] = uDvt.ptr<double>(0)[1];
            T[1][0] = uDvt.ptr<double>(1)[0];
            T[1][1] = uDvt.ptr<double>(1)[1];
            d[1] = temp;
        }
    }
    else
    {
        Mat D = (Mat_<double>(2, 2) << d[0], 0.0, 0.0, d[1]);
        Mat Dvt = D * vt;
        Mat uDvt = u * Dvt;
        T[0][0] = uDvt.ptr<double>(0)[0];
        T[0][1] = uDvt.ptr<double>(0)[1];
        T[1][0] = uDvt.ptr<double>(1)[0];
        T[1][1] = uDvt.ptr<double>(1)[1];
    }
    double var1 = 0.0;
    for (int i = 0; i < 5; i++)
        var1 += src_demean[i][0] * src_demean[i][0];
    var1 = var1 / 5;
    double var2 = 0.0;
    for (int i = 0; i < 5; i++)
        var2 += src_demean[i][1] * src_demean[i][1];
    var2 = var2 / 5;
    double scale = 1.0 / (var1 + var2) * (s.ptr<double>(0)[0] * d[0] + s.ptr<double>(1)[0] * d[1]);
    double TS[2];
    TS[0] = T[0][0] * src_mean[0] + T[0][1] * src_mean[1];
    TS[1] = T[1][0] * src_mean[0] + T[1][1] * src_mean[1];
    T[0][2] = dst_mean[0] - scale * TS[0];
    T[1][2] = dst_mean[1] - scale * TS[1];
    T[0][0] *= scale;
    T[0][1] *= scale;
    T[1][0] *= scale;
    T[1][1] *= scale;
    Mat transform_mat = (Mat_<double>(2, 3) << T[0][0], T[0][1], T[0][2], T[1][0], T[1][1], T[1][2]);
    return transform_mat;
};

static void alignCrop(InputArray _src_img, Yolov5Face_BoxStruct generate_boxes, OutputArray _aligned_img)
{
    float src_point[5][2];
    src_point[0][0] = generate_boxes.keypoint[0].x;
    src_point[0][1] = generate_boxes.keypoint[0].y;
    src_point[1][0] = generate_boxes.keypoint[1].x;
    src_point[1][1] = generate_boxes.keypoint[1].y;
    src_point[2][0] = generate_boxes.keypoint[2].x;
    src_point[2][1] = generate_boxes.keypoint[2].y;
    src_point[3][0] = generate_boxes.keypoint[3].x;
    src_point[3][1] = generate_boxes.keypoint[3].y;
    src_point[4][0] = generate_boxes.keypoint[4].x;
    src_point[4][1] = generate_boxes.keypoint[4].y;

    Mat warp_mat = getSimilarityTransformMatrix(src_point);
    warpAffine(_src_img, _aligned_img, warp_mat, Size(112, 112));
};

static void RoiFillter_face(vector<Yolov5Face_BoxStruct> &detectResults, std::vector<cv::Point> contours, float score)
{
    // std::string strs = jROI.toStyledString();
    // std::cout<<strs<<std::endl;

    int t = detectResults.size();

    for (int i = 0; i < t; i++) // 遍历原有的框框图，把符合roi的标准筛选出来
    {
        for (auto r_ = detectResults.begin(); r_ != detectResults.end(); r_++)
        {
            if (score > r_->score) // 分数过滤
            {
                detectResults.erase(r_);
                break;
            }
            if (contours.size() == 0)
                continue;

            cv::Point CenterPoint((r_->x1 + r_->x2) / 2, (r_->y1 + r_->y2) / 2);
            if (cv::pointPolygonTest(contours, CenterPoint, true) < 0) // 用于测试一个点是否在多边形中，contours是上面输入roi
            {
                detectResults.erase(r_);
                break;
            }
        }
    }
};

static void rectangleToSquare(float x1, float y1, float x2, float y2, cv::Rect& square) {
    // 计算宽度和高度
    float width = std::abs(x2 - x1);
    float height = std::abs(y2 - y1);
    // 正方形的边长取最大值
    float side_length = std::max(width, height);
    // 计算中心点
    float center_x = (x1 + x2) / 2.0f;
    float center_y = (y1 + y2) / 2.0f;
    // 计算正方形的四个顶点
    // float half_side = side_length / 2.2f;
    float half_side = side_length / 2.0f;
    square.x = center_x - half_side;
    square.y = center_y - half_side;
    square.width = half_side*2;
    square.height = half_side*2;
};

struct T_CurResult {
	cv::Mat img;
	std::vector<Yolov5Face_BoxStruct> result;
	int rtsp_size;
};



// 传入并原地在 face_result.img 上作画
static void draw_face_keypoints(T_CurResult& face_result,
                                bool draw_bbox = false,
                                bool draw_score = false)
{
    cv::Mat& img = face_result.img;
    if (img.empty()) return;

    const int radius = 3;                 // 关键点圆半径
    const int thickness = 2;              // 线条粗细
    const cv::Scalar ptColor(0, 255, 0);  // 关键点颜色（绿）
    const cv::Scalar lnColor(255, 0, 0);  // 连接线颜色（蓝）
    const cv::Scalar boxColor(0, 0, 255); // 框颜色（红）
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    const double fontScale = 0.5;

    for (const auto& det : face_result.result)
    {
        // 1) 画框（可选）
        if (draw_bbox) {
            cv::rectangle(img, cv::Rect(cv::Point(det.x1, det.y1),
                                        cv::Point(det.x2, det.y2)),
                          boxColor, 2);
        }

        // 2) 画关键点
        for (const auto& p : det.keypoint) {
            // 防越界（如果点在图外则跳过）
            if (p.x < 0 || p.y < 0 || p.x >= img.cols || p.y >= img.rows) continue;
            cv::circle(img, p, radius, ptColor, cv::FILLED, cv::LINE_AA);
        }

        // 3) 若为常见5点（0左眼,1右眼,2鼻尖,3左嘴角,4右嘴角），画连线增强可视化
        if (det.keypoint.size() >= 5) {
            auto P = det.keypoint; // 简写
            // 两眼连线
            cv::line(img, P[0], P[1], lnColor, thickness, cv::LINE_AA);
            // 眼到鼻
            cv::line(img, P[0], P[2], lnColor, 1, cv::LINE_AA);
            cv::line(img, P[1], P[2], lnColor, 1, cv::LINE_AA);
            // 鼻到嘴角
            cv::line(img, P[2], P[3], lnColor, 1, cv::LINE_AA);
            cv::line(img, P[2], P[4], lnColor, 1, cv::LINE_AA);
            // 嘴角连线
            cv::line(img, P[3], P[4], lnColor, 1, cv::LINE_AA);
        }

        // 4) 显示分数（可选）
        if (draw_score) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "s=%.2f", det.score);
            int tx = std::max(0, det.x1);
            int ty = std::max(0, det.y1 - 5);
            cv::putText(img, buf, cv::Point(tx, ty), fontFace, fontScale,
                        boxColor, 1, cv::LINE_AA);
        }
    }
}


static std::shared_ptr<faceYOLO> g_det_model = nullptr;
static std::mutex det_mutex;
static std::map<std::string, BlockingQueue<T_CurResult>> rtsp_result_list;
static int online_face_num; 



static T_CurResult GetResult(cv::Mat &img, const std::string &rtsp_path, int _rtsp_size)
{
	std::lock_guard<std::mutex> lock(det_mutex);
	T_CurResult tmp;
	if (rtsp_result_list[rtsp_path].IsEmpty()){
		tmp.result = g_det_model->infer(img);
		tmp.img = img;
		for (int i = 0; i < _rtsp_size - 1; i++){
			rtsp_result_list[rtsp_path].Push(tmp);
		}
	}else{
		rtsp_result_list[rtsp_path].Pop(tmp);
	}

	return tmp;
}


class faceRec
{
public:
//	std::shared_ptr<faceYOLO> det_model;
//    std::shared_ptr<faceRecognizer> rec_model;
    // cv::Point2f points_ref[4] = {
    //     cv::Point2f(0, 0),
    //     cv::Point2f(100, 0),
    //     cv::Point2f(100, 32),
    //     cv::Point2f(0, 32)};
	
    // ===== 单例入口 =====
    static faceRec& instance() {
        std::call_once(s_once_, [] { s_inst_.reset(new faceRec); });
        return *s_inst_;
    }

    // ===== 初始化（只执行一次，多次调用安全） =====
    void init(const std::string& model_path, int modelClassNum = 1) {
        (void)modelClassNum;
        std::call_once(init_once_, [&]{
            // 检测模型
            det_model_ = std::make_shared<faceYOLO>();
            det_model_->init("face_detection/det.pt", 1, 0);

            // 识别模型
            rec_model_ = std::make_shared<faceRecognizer>();
            rec_model_->init("face_detection/rec.pt", 1, 0);
        });
    }

    int infer(cv::Mat img, std::vector<cv::Point> contours, std::vector<Yolov5Face_BoxStruct> &face_result, std::vector<int> &delNums, std::vector<float> &input, int &querySize, float score, int limarea)
    {
        if (img.empty())
            return 0;
        auto start = std::chrono::steady_clock::now();
        {
			std::lock_guard<std::mutex> lk(det_mu_);
	        face_result = det_model_->infer(img);
		}
        auto end_det = std::chrono::steady_clock::now();

        RoiFillter_face(face_result, contours, score);
        querySize = 0;
        input.resize(0);
        delNums.resize(0);
        for (int i = 0; i < face_result.size(); i++)
        {
            // // 人脸质量检测
            int quality = face_quality(img, face_result[i], limarea);
            if (!quality)
            {
                delNums.push_back(i);
                continue;
            }

            // format人脸数据
            cv::Mat aligned_face1;
            alignCrop(img, face_result[i], aligned_face1);
            // 512维向量
            auto start_rec = std::chrono::steady_clock::now();
            vector<float> feature1;
            {
                std::lock_guard<std::mutex> lk(rec_mu_);
                feature1 = rec_model_->infer(aligned_face1);
            }
            auto end_rec = std::chrono::steady_clock::now();
            // cout << "[DEBUG] time infer rec: "  << std::chrono::duration_cast<std::chrono::milliseconds>(end_rec - start_rec).count() << endl;

            for (int i = 0; i < 512; i++)
            {
                input.push_back(feature1[i]);
            }
            querySize++;
        }
        auto end = std::chrono::steady_clock::now();
        // cout << "[DEBUG] time infer det: " << " " << std::chrono::duration_cast<std::chrono::milliseconds>(end_det - start).count() << " ms, all: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << endl;
        return 1;
    }

	int infer(cv::Mat img,
			  const std::vector<cv::Point> contours,
			  std::vector<Yolov5Face_BoxStruct> &face_result,
			  std::vector<faceInfo>& out_faces,   // 引用输出
			  float score,
			  int limarea)
	{
		if (img.empty()) return 0;
	
        auto start = std::chrono::steady_clock::now();
        {
			std::lock_guard<std::mutex> lk(det_mu_);
	        face_result = det_model_->infer(img);
		}
        auto end_det = std::chrono::steady_clock::now();
	
		// ROI过滤
        RoiFillter_face(face_result, contours, score);
	
		int added = 0;
		cv::Mat aligned_face1;			// 循环外复用，减少重复分配
		std::vector<float> feature1; 	// 循环外复用
	
		for (size_t i = 0; i < face_result.size(); ++i) {
			// 1) 质量过滤
			if (!face_quality(img, face_result[i], limarea)) {
				continue;
			}
	
            alignCrop(img, face_result[i], aligned_face1);
            // 512维向量
            auto start_rec = std::chrono::steady_clock::now();
            vector<float> feature1;
            {
                std::lock_guard<std::mutex> lk(rec_mu_);
                feature1 = rec_model_->infer(aligned_face1);
            }
            auto end_rec = std::chrono::steady_clock::now();
			// 可按需打印耗时：
			// std::cout << "[DEBUG] rec ms: "
			//			 << std::chrono::duration_cast<std::chrono::milliseconds>(t_rec1 - t_rec0).count() << std::endl;
	
			if (feature1.size() != 512) {
				// 模型输出异常，跳过该人脸
				continue;
			}
	
			// 4) 组织输出
			const auto& b = face_result[i];
			faceInfo tmp;
			tmp.x = b.x1;
			tmp.y = b.y1;
			tmp.w = b.x2 - b.x1;
			tmp.h = b.y2 - b.y1;
			tmp.score = b.score;
			tmp.conf = 0.0f;
			tmp.id = -2;
			tmp.fileName.clear();
	
			// 512特征值
			tmp.feature.clear();
			tmp.feature.insert(tmp.feature.end(), feature1.begin(), feature1.end());
	
			out_faces.push_back(std::move(tmp));
			++added;
		}
	
		auto t1 = std::chrono::steady_clock::now();
		// std::cout << "[DEBUG] det ms: "
		//			 << std::chrono::duration_cast<std::chrono::milliseconds>(t_det_end - t0).count()
		//			 << ", all ms: "
		//			 << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
		//			 << std::endl;
	
		return added; // 返回新增的人脸数量，比固定返回1更有用
	}

#if 0    
	~faceRec()
	{
		std::lock_guard<std::mutex> lock(det_mutex);
		online_face_num--;
		if (online_face_num == 0 && !g_det_model){
			g_det_model.reset();
		}
	}

    void init(std::string model_path, int modleClassNum = 1)
    {
    	if (model_path == "faceserver"){
			det_model = std::make_shared<faceYOLO>();
			det_model->init("face_detection/det.pt", 1, 0);
		}
    	{
	        std::lock_guard<std::mutex> lock(det_mutex);
	        if (!g_det_model) {  // 确保 det_model 只初始化一次
	            g_det_model = std::make_shared<faceYOLO>();
	            g_det_model->init("face_detection/det.pt", 1, 0);
	        }
			online_face_num++;
		}
        rec_model = std::make_shared<faceRecognizer>();
        rec_model->init("face_detection/rec.pt", 1, 0);
    }

    int infer(cv::Mat img, std::vector<cv::Point> contours, T_CurResult &face_result, std::vector<int> &delNums, std::vector<float> &input, int &querySize, float score, int limarea, const std::string &_rtsp_path)
    {
        if (img.empty())
            return 0;
        auto start = std::chrono::steady_clock::now();
        face_result = GetResult(img, _rtsp_path, face_result.rtsp_size);
        auto end_det = std::chrono::steady_clock::now();

        RoiFillter_face(face_result.result, contours, score);
        querySize = 0;
        input.resize(0);
        delNums.resize(0);
        for (int i = 0; i < face_result.result.size(); i++)
        {
            // // 人脸质量检测
            int quality = face_quality(face_result.img, face_result.result[i], limarea);
            if (!quality)
            {
                delNums.push_back(i);
                continue;
            }

            // format人脸数据
            cv::Mat aligned_face1;
            alignCrop(face_result.img, face_result.result[i], aligned_face1);
            // 512维向量
            auto start_rec = std::chrono::steady_clock::now();
            vector<float> feature1 = rec_model->infer(aligned_face1);
            auto end_rec = std::chrono::steady_clock::now();
            // cout << "[DEBUG] time infer rec: "  << std::chrono::duration_cast<std::chrono::milliseconds>(end_rec - start_rec).count() << endl;

//            for (int i = 0; i < 512; i++)
//            {
//                input.push_back(feature1[i]);
//            }
//            querySize++;
			if (feature1.size() == 512){
				input.insert(input.end(), feature1.begin(), feature1.end());
				++querySize;
			}
        }
        auto end = std::chrono::steady_clock::now();
        // cout << "[DEBUG] time infer det: " << " " << std::chrono::duration_cast<std::chrono::milliseconds>(end_det - start).count() << " ms, all: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << endl;
        return 1;
    }
	
	int infer(cv::Mat img,
			  const std::vector<cv::Point> contours,
			  T_CurResult& face_result,
			  std::vector<faceInfo>& out_faces,   // 引用输出
			  float score,
			  int limarea,
			  const std::string& _rtsp_path)
	{
		if (img.empty()) return 0;
	
		auto t0 = std::chrono::steady_clock::now();
	
		// 推理检测
		face_result = GetResult(img, _rtsp_path, face_result.rtsp_size);
		auto t_det_end = std::chrono::steady_clock::now();
	
		// ROI过滤
		RoiFillter_face(face_result.result, contours, score);
	
		int added = 0;
		cv::Mat aligned_face1;			// 循环外复用，减少重复分配
		std::vector<float> feature1; 	// 循环外复用
	
		for (size_t i = 0; i < face_result.result.size(); ++i) {
			// 1) 质量过滤
			if (!face_quality(face_result.img, face_result.result[i], limarea)) {
				continue;
			}
	
            alignCrop(face_result.img, face_result.result[i], aligned_face1);
            // 512维向量
            auto start_rec = std::chrono::steady_clock::now();
            vector<float> feature1 = rec_model->infer(aligned_face1);
            auto end_rec = std::chrono::steady_clock::now();
			// 可按需打印耗时：
			// std::cout << "[DEBUG] rec ms: "
			//			 << std::chrono::duration_cast<std::chrono::milliseconds>(t_rec1 - t_rec0).count() << std::endl;
	
			if (feature1.size() != 512) {
				// 模型输出异常，跳过该人脸
				continue;
			}
	
			// 4) 组织输出
			const auto& b = face_result.result[i];
			faceInfo tmp;
			tmp.x = b.x1;
			tmp.y = b.y1;
			tmp.w = b.x2 - b.x1;
			tmp.h = b.y2 - b.y1;
			tmp.score = b.score;
			tmp.conf = 0.0f;
			tmp.id = -2;
			tmp.fileName.clear();
	
			// 512特征值
			tmp.feature.clear();
			tmp.feature.insert(tmp.feature.end(), feature1.begin(), feature1.end());
	
			out_faces.push_back(std::move(tmp));
			++added;
		}
	
		auto t1 = std::chrono::steady_clock::now();
		// std::cout << "[DEBUG] det ms: "
		//			 << std::chrono::duration_cast<std::chrono::milliseconds>(t_det_end - t0).count()
		//			 << ", all ms: "
		//			 << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
		//			 << std::endl;
	
		return added; // 返回新增的人脸数量，比固定返回1更有用
	}
#endif

    int infer(cv::Mat img, std::vector<Yolov5Face_BoxStruct> &face_result, std::vector<float> &input)
    {

        face_result = det_model_->infer(img, 0.3);
        if (face_result.size() <= 0)
        {
            cout << "[ERROR] face count is 0" << endl;
            return 0;
        }

        // // 人脸质量检测
        int quality = face_quality(img, face_result[0], 80);

        if (!quality)
        {
            return 2;
        }

        // format人脸数据
        cv::Mat aligned_face1;
        alignCrop(img, face_result[0], aligned_face1);
        // 512维向量
        vector<float> feature1 = rec_model_->infer(aligned_face1);

        for (int i = 0; i < 512; i++)
        {
            input.push_back(feature1[i]);
        }

        return 1;
    }


    // 禁止拷贝/移动
    faceRec(const faceRec&) = delete;
    faceRec& operator=(const faceRec&) = delete;
    faceRec(faceRec&&) = delete;
    faceRec& operator=(faceRec&&) = delete;

private:
    faceRec() = default;

    // 共享模型（仅此单例持有）
    std::shared_ptr<faceYOLO>      det_model_;
    std::shared_ptr<faceRecognizer> rec_model_;

    // 推理互斥：把真正的引擎调用包起来，粒度尽量小
    std::mutex det_mu_;
    std::mutex rec_mu_;

    // 初始化只执行一次
    std::once_flag init_once_;

    // 单例
    static std::unique_ptr<faceRec> s_inst_;
    static std::once_flag           s_once_;	
};

// 静态成员定义
inline std::unique_ptr<faceRec> faceRec::s_inst_;
inline std::once_flag           faceRec::s_once_;


#if defined X86_TS
class faceRecExpression
{
public:
    std::shared_ptr<faceYOLO> det_model;
    std::shared_ptr<faceRecognizer> rec_model;
    std::shared_ptr<faceExpression> fer_model;
    // cv::Point2f points_ref[4] = {
    //     cv::Point2f(0, 0),
    //     cv::Point2f(100, 0),
    //     cv::Point2f(100, 32),
    //     cv::Point2f(0, 32)};

    void init(std::string model_path, int modleClassNum = 1)
    {
        det_model = std::make_shared<faceYOLO>();
        det_model->init("facial_expressions_detection/det.pt", 1, 0);
        rec_model = std::make_shared<faceRecognizer>();
        rec_model->init("facial_expressions_detection/rec.pt", 1, 0);
        fer_model = std::make_shared<faceExpression>();
        fer_model->init("facial_expressions_detection/fer.pt", 1, 0);
    }

    int infer(cv::Mat img, std::vector<cv::Point> contours, std::vector<Yolov5Face_BoxStruct> &face_result, std::vector<int> &delNums, std::vector<float> &input, int &querySize, float score, int limarea, std::vector<int> &fer_results)
    {
        if (img.empty())
            return 0;
        auto start = std::chrono::steady_clock::now();
        face_result = det_model->infer(img);
        auto end_det = std::chrono::steady_clock::now();

        RoiFillter_face(face_result, contours, score);
        querySize = 0;
        input.resize(0);
        delNums.resize(0);
        for (int i = 0; i < face_result.size(); i++)
        {
            // // 人脸质量检测
            int quality = face_quality(img, face_result[i], limarea);
            if (!quality)
            {
                fer_results.push_back(0); //默认
                delNums.push_back(i);
                continue;
            }

            // format人脸数据
            cv::Mat aligned_face1;
            alignCrop(img, face_result[i], aligned_face1);
            // 512维向量
            auto start_rec = std::chrono::steady_clock::now();
            vector<float> feature1 = rec_model->infer(aligned_face1);
            // cv::resize(aligned_face1, aligned_face1, cv::Size(64, 64));
            // int fer_result = fer_model->infer(aligned_face1);
            // DAN
            cv::Rect rect;
		    rectangleToSquare(face_result[i].x1, face_result[i].y1, face_result[i].x2, face_result[i].y2, rect);
            if (rect.x < 0) rect.x = 0;
            if (rect.y < 0) rect.y = 0;
            if (rect.x + rect.width > img.cols) rect.width = img.cols - rect.x;
            if (rect.y + rect.height > img.rows) rect.height = img.rows - rect.y;
            cv::Mat aligned_face3=img(rect);
            int fer_result = fer_model->infer(aligned_face3);
            fer_results.push_back(fer_result);
            auto end_rec = std::chrono::steady_clock::now();
            // cout << "[DEBUG] time infer rec: "  << std::chrono::duration_cast<std::chrono::milliseconds>(end_rec - start_rec).count() << endl;

            for (int i = 0; i < 512; i++)
            {
                input.push_back(feature1[i]);
            }
            querySize++;
        }
        auto end = std::chrono::steady_clock::now();
        // cout << "[DEBUG] time infer det: " << " " << std::chrono::duration_cast<std::chrono::milliseconds>(end_det - start).count() << " ms, all: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << endl;
        return 1;
    }

    int infer(cv::Mat img, std::vector<Yolov5Face_BoxStruct> &face_result, std::vector<float> &input)
    {

        face_result = det_model->infer(img, 0.3);
        if (face_result.size() <= 0)
        {
            cout << "[ERROR] face count is 0" << endl;
            return 0;
        }

        // // 人脸质量检测
        int quality = face_quality(img, face_result[0], 80);

        if (!quality)
        {
            return 2;
        }

        // format人脸数据
        cv::Mat aligned_face1;
        alignCrop(img, face_result[0], aligned_face1);
        // 512维向量
        vector<float> feature1 = rec_model->infer(aligned_face1);

        for (int i = 0; i < 512; i++)
        {
            input.push_back(feature1[i]);
        }

        return 1;
    }
};
#endif
