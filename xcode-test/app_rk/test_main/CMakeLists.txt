cmake_minimum_required(VERSION 3.14)

project(one)


# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
if(CMAKE_BUILD_TYPE MATCHES Debug)
add_definitions(-DDEED_DEBUG) # 调试(不用lic)，发表用release
message(STATUS "Debug...")
endif()


add_definitions(-std=c++14)
add_definitions(-DAPI_EXPORTS)



#必选开关，编译http库的时候，选用poll模型，启用录像会把select(跨操作系统平台)最大数耗尽,window不支持poll
add_definitions(-DCPPHTTPLIB_USE_POLL)
add_definitions(-DCROW_MAIN)


#平台选择
add_definitions(-DRK)
#add_definitions(-DUSE_LANDMARK) # 跨车位用关键点
add_definitions(-DUSE_CROSS) # 跨车位跟车牌检测区别
# add_definitions(-DNeedRecord) #录像
# add_definitions(-DDECNOTRESIZE) #吸烟等要原图
# add_definitions(-DJETSON808) # 用"cat /sys/class/net/e*/address" 取第一个地址


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wfatal-errors -w  -O0 -D_MWAITXINTRIN_H_INCLUDED")# -fsanitize=address -fno-omit-frame-pointer")


#rk
include(${CMAKE_SOURCE_DIR}/infer/rknn.cmake)
include(${CMAKE_SOURCE_DIR}/../common/common.cmake)
include_directories(${CMAKE_SOURCE_DIR}/../post/inferModel)


link_libraries(stdc++ pthread opencv_imgproc  opencv_core opencv_highgui opencv_imgcodecs opencv_video opencv_videoio 
rockchip_mpp
crypt
rga
crypto
ssl
)
link_libraries(avcodec avformat avdevice avutil swresample swscale )



# # 1 解码ffmpeg
set(main ${CMAKE_SOURCE_DIR}/../post/algsCom/main_all.cpp)
add_executable(one  ${main}  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp ${common}  ${track} ${rknn_src} ${CMAKE_SOURCE_DIR}/prefer/get_videoffmpeg.cpp)





