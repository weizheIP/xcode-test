# 测试环境

## 测试设备

### rk

192.168.3.62：AIOT平台

### jetson

192.168.3.72 20T
192.168.3.73 40T

- 演示设备

汇总平台

https://cbi_new.cvvii.com/
13012345678
123456

安防设备2：
外部访问：http://afcs2.cvvii.com:38080/
内部访问：http://192.168.3.41:9696
账号：civi，密码：123456

养殖设备：
外部访问：http://yzcs.cvvii.com:38080/
内部访问：http://192.168.3.51:9696
账号：civi，密码：123456

## 测试相机

TL-IPC43K-4 2.0
192.168.3.4 办公西 admin zkds1688
192.168.3.69 办公东
192.168.3.202 展区1
192.168.3.32 展区2
192.168.3.3 前台

rtsp://admin:civicint1110@192.168.3.160:554/Streaming/Channels/101
rtsp://admin:123456@192.168.3.161:554/ch01.264
rtsp://admin:civicint1110@192.168.3.170:554/Streaming/Channels/101
rtsp://admin:HuaWei123@192.168.3.163:554/LiveMedia/ch1/Media1
rtsp://admin:HuaWei123@192.168.3.164:554/LiveMedia/ch1/Media1
rtsp://admin:HuaWei123@192.168.3.165:554/LiveMedia/ch1/Media1
rtsp://admin:HuaWei123@192.168.3.166:554/LiveMedia/ch1/Media1

rtsp://admin:zkds1688@192.168.3.3:554/stream1

rtsp://admin:zkds1688@192.168.3.4:554/stream1

# SDK代码

整体架构：

1. 相机：一次解码，APP，每个相机线程根据绑定的算法 创建3业务后处理线程 (N)
2. 算法：把模型相关的都放在算法类里面 Algs_APP ，遍历算法模型绑定的相机，进行推理 (1)
3. 业务后处理：由于一个算法会绑定多个相机，所以不能把模型放在后处理里面，避免每个实例都载入模型；业务每个相机都要有一个因为参数不同 (N)

## common

平台无关

```
http webosocket
encryption   加密
license 生成
apt install libssl-dev
apt install libboost-all-dev   20.04:1.71.0

```

业务主逻辑

业务相关的后处理：rk，x86和jetson是一样的

跟踪

opencv

librealsense

libmodbus

## 模型

每个模型只加载一次

多卡自动按顺序分配

## 应用交互

启动脚本 /civi/algs/start1.sh

算法主程序 /civi/algs/main/

算法子插件 /civi/algs/alg_id(如：fire_detection)/

lic读取应用../civiapp/licens/license.lic

## 应用API接口

获取算法参数："http://127.0.0.1:9696/plg/" + alg + "/calc/rtsp"

获取预览的地址：http://127.0.0.1:9696/api/camera/preview/rtsp

获取算法识别框的地址：http://127.0.0.1:9696/api/ws/getPushVideo

ws://127.0.0.1:9698/ws

更新相机状态：Post /api/camera/status

- 通用

布防时间work_time，目标持续时间target_continued_time，告警时间间隔time，告警阈值score，智能过滤auto_filter

```
// http://192.168.18.60:9696/plg/intrusion_detection/calc/rtsp
{
    "code": 0,
    "msg": "success",
    "data": [  
        {
            "camera_id": "6641cbd8e2f4060030c1e7cd",
            "rtsp": "rtsp://admin:HuaWei123@192.168.18.163:554/LiveMedia/ch1/Media1",
            "roi": [],
            "config": {
                "score": 0.7,
                "time": 360,
                "target_continued_time": 1,
                "mode": 0,
                "video": 0,
                "alarm_sound_column_ip": "",  
                "work_time": [  
                    [
                        "00:00:00",
                        "23:59:59"
                    ]
                ],
                "auto_filter": 1,
                "alarm_level": 3
            }
        }
    ]
}
```

# 开发部署

## x86

开发机：192.168.3.20 ls 123456

```
docker run -itd -e TZ="Asia/Shanghai" --restart=always --privileged -v /civi/:/civi -v /etc/localtime:/etc/localtime --gpus all --ipc=host --net=host --name=test registry.cn-hangzhou.aliyuncs.com/civicint/civi_image_c /bin/bash
```

## rk

debina11

系统自带mpp库、rga和runtime库

ffmpeg可以用带有mpp的版本

```
apt install libopencv-dev   （debina11里面默认是4.5.1）

apt install libavformat-dev  libavcodec-dev libavformat-dev libavutil-dev libswresample-dev libswscale-dev
```

opencv需要的库liblapack-dev  libatlas-base-dev libjpeg-dev已经自带，除非用容器才要自己装

build-essential

线上版本容器

ubuntu:20.04

ffmpeg_mpp 5.1

```
docker run -it --restart=always -e TZ="Asia/Shanghai" --privileged -v /userdata/civi:/civi --net=host --ipc=host --name=test registry.cn-hangzhou.aliyuncs.com/civicint/civi_image_rk_c:latest  /bin/bash
```

- 交叉编译

```
/media/ps/data1/liuym/chipsdk/rockchip/zip/gcc-buildroot-9.3.0-2020.03-x86_64_aarch64-rockchip-linux-gnu-firefly
```

## jetson

镜像：nvcr.io/nvidia/l4t-ml:r35.2.1-py3  nvcr.io/nvidia/l4t-ml:r32.7.1-py3

gstreamer、opencv4.5.1和tensorrt 都用容器自带,boost1.71.0

gcc9.4 ubuntu20.04

jepack 6.2: nvcr.io/nvidia/l4t-ml:r36.4.3-py3

jepack 6.1: 36.4 Ubuntu 22.04

```
docker run -it --restart=always --privileged --runtime nvidia --network host -v /civi:/civi  --ipc=host --name=civi_algs nvcr.io/nvidia/l4t-ml:r32.7.1-py3 bash
```
