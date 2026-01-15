cmake_minimum_required(VERSION 3.10)

project(one)


set(OPENCV_ENABLE)
# 
add_definitions(-std=c++14)
add_definitions(-DAPI_EXPORTS)
add_definitions(-DCROW_MAIN)


# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)




#必选开关，编译http库的时候，选用poll模型，启用录像会把select最大数耗尽,window不支持poll
add_definitions(-DCPPHTTPLIB_USE_POLL)


#功能设置
add_definitions(-DJETSON)
# add_definitions(-DDECNOTRESIZE) #吸烟等要原图
# add_definitions(-DNeedRecord) #录像
add_definitions(-DJETSON808) # 808 70T的用"cat /sys/class/net/e*/address" 取第一个地址


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wfatal-errors -w  -O0 -D_MWAITXINTRIN_H_INCLUDED")# -fsanitize=address -fno-omit-frame-pointer")

# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64) # nx21t用这个

find_package(OpenCV REQUIRED)

include(${CMAKE_SOURCE_DIR}/../common/common.cmake)



include_directories(
  # /usr/local/include/opencv4
  ${CMAKE_SOURCE_DIR}/infer/v8
)
set(infer_mask ${CMAKE_SOURCE_DIR}/infer/v8/yolo.cpp
${CMAKE_SOURCE_DIR}/infer/v8/muti.cu)


link_directories(
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_jetson/lib
)

link_libraries(stdc++ pthread
# opencv_imgproc  opencv_core opencv_highgui opencv_imgcodecs opencv_video opencv_videoio opencv_dnn
${OpenCV_LIBS}
# avcodec
# avformat
# avutil
# swresample
# swscale
# boost_thread
crypto
#cublas curand cudart
cuda  cudnn 
 nvinfer 

)


set(main ${CMAKE_SOURCE_DIR}/../post/algsCom/main_all.cpp)
cuda_add_executable(one ${main} ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp   ${infer_mask}    ${common} ${track})

#cuda_add_executable(test /civi/yuqk/code/oneDecode/test.cpp ${infer_mask} ${muxer}  ${count} ${post} ${common} ${track})


# cuda_add_executable(main_count  ${cdsServer} ${infer_mask}   ${common})


# add_executable(test ${CMAKE_SOURCE_DIR}/test/opencv_gst_nvdec.cpp)
