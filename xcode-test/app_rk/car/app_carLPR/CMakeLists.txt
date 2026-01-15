cmake_minimum_required(VERSION 3.14)

project(trunk)


#set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wfatal-errors -w  -O0")# -fsanitize=address -fno-omit-frame-pointer")

add_definitions(-DCPPHTTPLIB_USE_POLL)
add_definitions(-DCROW_MAIN)
add_definitions(-DAPI_EXPORTS)

#平台选择
add_definitions(-DRK)
add_definitions(-DDECNOTRESIZE) # 要原图
add_definitions(-DUSE_LANDMARK) # 关键点


#rk
include(${CMAKE_SOURCE_DIR}/infer/rknn.cmake)
include(${CMAKE_SOURCE_DIR}/../common/common.cmake)


set(car_infer
${CMAKE_SOURCE_DIR}/infer/car_plate/car_infer.cpp 
${CMAKE_SOURCE_DIR}/infer/car_plate/postprocess.cc
)
set(prefer ${CMAKE_SOURCE_DIR}/prefer/get_videoffmpeg.cpp)

set(main ${CMAKE_SOURCE_DIR}/../post/car/app_carLPR/main.cpp)
set(post_process ${CMAKE_SOURCE_DIR}/../post/car/app_carLPR/post_process.cpp)



link_libraries(stdc++ pthread opencv_imgproc  opencv_core opencv_highgui opencv_imgcodecs opencv_video opencv_videoio 
rockchip_mpp
opencv_dnn
opencv_calib3d
rknn_api
avcodec
avformat
avutil
swresample
swscale
boost_thread
crypt
rga
crypto
ssl
)

add_executable(carLPR ${main} ${muxer} ${app} ${car_infer} ${infer2} ${oc_sort} ${post_process} ${prefer}  ${post2} ${common} ${rknn_src} ${track})


