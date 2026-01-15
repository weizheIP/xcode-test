#include "YOLOv8LibTorch.h" //放在所有头文件最前面，避免和opencv冲突出错

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <float.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include "BYTETracker.h"




int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [videopath]\n", argv[0]);
        return -1;
    }

    torch::jit::script::Module module = torch::jit::load("/media/ps/data1/liuym/202012-count/ultralytics/runs/detect/car/1027s5800-bad/weights/best.torchscript",torch::kCUDA);


    const char* videopath = argv[1];  //"/media/ps/data2/civi/share/智慧安防/点点车牌/大型车遮挡/0110/e12775f0-ae15-11ee-836f-1bf5db1196bd.mp4"

    cv::VideoCapture cap(videopath);
	if (!cap.isOpened())
		return 0;

	// int img_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	// int img_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    long nFrame = static_cast<long>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    cout << "Total frames: " << nFrame << " fps: " << fps << endl;

    cv::VideoWriter writer("demo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));

    cv::Mat frame;
    BYTETracker tracker(fps, 500);
    int num_frames = 0;
    int total_ms = 1;
	for (;;)
    {
        if(!cap.read(frame))
            break;
        num_frames ++;
        if (num_frames % 20 == 0)
        {
            cout << "Processing frame " << num_frames << " (" << num_frames * 1000000 / total_ms << " fps)" << endl;
        }
		if (frame.empty())
			break;
        auto start = chrono::system_clock::now();

        cv::Mat img = frame.clone();
        std::vector<float> pad_info = LetterboxImage(img, img, cv::Size(800, 800));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte).cuda();
        imgTensor = imgTensor.permute({2, 0, 1});
        imgTensor = imgTensor.toType(torch::kFloat);
        imgTensor = imgTensor.div(255);
        imgTensor = imgTensor.unsqueeze(0);
        std::cout << "1111" << std::endl;
        torch::Tensor preds = module.forward({imgTensor}).toTensor();
        std::vector<Detection> output = PostProcessing(preds.permute({0, 2, 1}), pad_info[0], pad_info[1], pad_info[2], frame.size(), 0.1, 0.6)[0];
        std::vector<Object> objects;
        for (int i = 0; i < output.size(); ++i) {
            Detection detection = output[i];
            Object object;
            object.rect=detection.bbox;
            object.label=detection.class_idx;
            object.prob=detection.score;
            objects.push_back(object);
        }             


        vector<STrack> output_stracks = tracker.update(objects); // 检测的输出不要conf过滤，用0.1
        auto end = chrono::system_clock::now();
        total_ms = total_ms + chrono::duration_cast<chrono::microseconds>(end - start).count();
        for (int i = 0; i < output_stracks.size(); i++)
		{
			vector<float> tlwh = output_stracks[i].tlwh;
			bool vertical = tlwh[2] / tlwh[3] > 1.6;
			if (tlwh[2] * tlwh[3] > 20 && !vertical)
			{
				cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
				cv::putText(frame, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5), 
                        0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
			}
		}
        cv::putText(frame, cv::format("frame: %d fps: %d", num_frames, num_frames * 1000000 / total_ms), 
                cv::Point(0, 30), 0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        writer.write(frame);
        // cv::imshow("VideoOD", frame);
        char c = cv::waitKey(1);
        if (c > 0)
        {
            break;
        }
    }
    cap.release();
    cout << "FPS: " << num_frames * 1000000 / total_ms << endl;

    return 0;
}
