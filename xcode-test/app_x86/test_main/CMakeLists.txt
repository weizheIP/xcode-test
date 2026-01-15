cmake_minimum_required(VERSION 3.14)

project(oneDecode)

add_definitions(-DAPI_EXPORTS)
#必选开关，编译http库的时候，选用poll模型，启用录像会把select最大数耗尽,window不支持poll
add_definitions(-DCPPHTTPLIB_USE_POLL)
add_definitions(-DCROW_MAIN)


#功能设置
# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

# option(X86_TS "Enable TS support" ON) # CMakeCache.txt 里已经缓存要删掉
# option(X86_TS "Enable TS support" OFF)
# option(X86_TENSORRT "Enable TensorRT support" ON)
# option(X86_TENSORRT "Enable TensorRT support" OFF)
set(X86_TS ON)
if(X86_TENSORRT)
    add_definitions(-DX86_TENSORRT)
endif()
if(X86_TS)
    add_definitions(-DX86_TS) # rtx2060
endif()

add_definitions(-DZHIWEI_GONGKONG) # 智微的工控机的用"cat /sys/class/net/e*/address"

# add_definitions(-DDECNOTRESIZE) #吸烟等要原图，注意会导致定制算法里面都要改ROI，ROI基于preview
# add_definitions(-DNeedRecord) #录像


if(CMAKE_BUILD_TYPE MATCHES Debug)
add_definitions(-DDEED_DEBUG) # 调试(不用lic)，发表用release
message(STATUS "Debug...")
endif()




add_definitions(-std=c++14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wfatal-errors -w  -O0 -D_MWAITXINTRIN_H_INCLUDED")# -fsanitize=address -fno-omit-frame-pointer")

include(${CMAKE_SOURCE_DIR}/infer/tensorrt.cmake)
include(${CMAKE_SOURCE_DIR}/../common/common.cmake)

include_directories(
  ${CMAKE_SOURCE_DIR}/../post/inferModel
)

link_libraries(stdc++ pthread 
opencv_imgproc  opencv_core opencv_highgui opencv_imgcodecs opencv_video opencv_videoio 
opencv_dnn
avcodec
avformat
avutil
swresample
swscale
boost_thread
crypto 
)

# # 11 preview
# set(main ${CMAKE_SOURCE_DIR}/../post/algsCom/main_preview.cpp)
# add_executable(one_preview  ${main}  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp ${common} ${CMAKE_SOURCE_DIR}/prefer/get_videoffmpeg.cpp) #gst有些奇怪的流放不了
# # add_executable(test_preview  ${CMAKE_SOURCE_DIR}/../post/algsCom/test_preview_opencv.cpp  ${common}  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp ) 

# # # 12 main_one
# # 巡栏点数推流,ts
# add_executable(one_xlds   ${CMAKE_SOURCE_DIR}/../post/algsCom/main_one.cpp  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp ${common}  ${track}  ${CMAKE_SOURCE_DIR}/prefer/get_videoffmpeg.cpp)
# # # 巡栏估重推流
# # add_executable(one_xlgz   ${CMAKE_SOURCE_DIR}/../post/algsCom/main_one.cpp  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp ${common}  ${track}  ${CMAKE_SOURCE_DIR}/prefer/get_videoffmpeg.cpp)


# 1 通用
set(main ${CMAKE_SOURCE_DIR}/../post/algsCom/main_all.cpp)
# set(main ${CMAKE_SOURCE_DIR}/../post/algsCom/main_one.cpp)
if(CMAKE_BUILD_TYPE MATCHES Debug)
add_executable(one_debug  ${main}  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp   ${common}  ${track})
else()
# add_executable(onegst  ${main}  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp  ${common}  ${track}) #gst大于30+会卡住，要加延迟可以50+路
# add_executable(oneimg  ${CMAKE_SOURCE_DIR}/../post/algsCom/main_all_img.cpp  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp  ${common}  ${track} ${CMAKE_SOURCE_DIR}/prefer/get_videoffmpeg.cpp) #图片流
add_executable(one  ${main}  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp  ${common}  ${track} ${CMAKE_SOURCE_DIR}/prefer/get_videoffmpeg.cpp) #ffmpeg一路300M，路数多了显存不够，加延迟
endif()

# # 2 停车/充电
# include_directories(/media/ps/data1/liuym/chipsdk/onnx/onnxruntime-linux-x64-gpu-1.12.1/include)
# link_directories(/media/ps/data1/liuym/chipsdk/onnx/onnxruntime-linux-x64-gpu-1.12.1/lib)
# link_libraries(onnxruntime)
# add_definitions(-DUSE_LANDMARK)
# add_definitions(-DUSE_CROSS) # 跨车位跟车牌检测区别
# include_directories(
# ${CMAKE_SOURCE_DIR}/infer/car_plate
# )
# file(GLOB car_infer ${CMAKE_SOURCE_DIR}/infer/car_plate/CrnnNet.cpp)
# set(main ${CMAKE_SOURCE_DIR}/../post/algsCom/car_charge/main_all.cpp)
# if(CMAKE_BUILD_TYPE MATCHES Debug)
# add_executable(one_car_debug  ${main}  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp   ${common} ${car_infer} ${track})
# else()
# add_executable(one_car  ${main}  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp  ${common} ${car_infer} ${track})
# endif()

# # 3 测试图片接口、南宁影子手机点数接口
# # add_executable(main_server_img  ${CMAKE_SOURCE_DIR}/../post/count/dbgz/main_server_img.cpp  ${common}) #ts
# add_executable(main_server_img  ${CMAKE_SOURCE_DIR}/../post/count/dbgz/main_server_img.cpp  ${common}  ${CMAKE_SOURCE_DIR}/infer/tensorrt/tr.cpp )

# 4 test 测试
# add_executable(test  ${CMAKE_SOURCE_DIR}/../post/algsCom/test.cpp  ${common})




# export PATH=/usr/local/cuda/bin/:$PATH

