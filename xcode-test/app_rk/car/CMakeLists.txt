cmake_minimum_required(VERSION 3.14)

project(car)

# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")

add_definitions(-std=c++14)
add_definitions(-DAPI_EXPORTS)


#平台选择
add_definitions(-DRK)
add_definitions(-DUSE_FACE)
add_definitions(-DUSE_LANDMARK) # 跨车位用关键点
add_definitions(-DDECNOTRESIZE) # 人脸要，传原图和1280
# add_definitions(-DSTRANGERDB)

#必选开关，编译http库的时候，选用poll模型，启用录像会把select(跨操作系统平台)最大数耗尽,window不支持poll
add_definitions(-DCPPHTTPLIB_USE_POLL)
add_definitions(-DCROW_MAIN)


#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -g -D_MWAITXINTRIN_H_INCLUDED")# -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wfatal-errors -w  -O0 -D_MWAITXINTRIN_H_INCLUDED")# -fsanitize=address -fno-omit-frame-pointer")


#rk
include(${CMAKE_SOURCE_DIR}/infer/rknn.cmake)
include(${CMAKE_SOURCE_DIR}/../common/common.cmake)

# faiss
include_directories(${CMAKE_SOURCE_DIR}/face
${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/faiss-1.7.4/
)
link_directories(
${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/faiss-1.7.4/build/faiss
)


set(face_infer
${CMAKE_SOURCE_DIR}/infer/face/face_recognition.cpp
${CMAKE_SOURCE_DIR}/infer/face/yolov5face/face2yolov5.cpp
${CMAKE_SOURCE_DIR}/infer/face/yolov5face/postprocess.cc
)

set(car_infer
${CMAKE_SOURCE_DIR}/infer/car_plate/car_infer.cpp 
${CMAKE_SOURCE_DIR}/infer/car_plate/postprocess.cc
)


link_libraries(stdc++ pthread opencv_imgproc  opencv_core opencv_highgui opencv_imgcodecs opencv_video opencv_videoio opencv_dnn opencv_calib3d
rknn_api
avcodec avformat avutil swresample swscale
boost_thread crypt crypto ssl
faiss
)

# # 旧
# file(GLOB main ${CMAKE_SOURCE_DIR}/../post/face/*.cpp)
# set(prefer ${CMAKE_SOURCE_DIR}/prefer/get_videoffmpeg.cpp)
# add_executable(faceServer ${main} ${prefer} ${face_infer} ${face_infer2}  ${common} ${track})


# 新
set(post ${CMAKE_SOURCE_DIR}/../post/face/face_db.cpp  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp )
file(GLOB main ${CMAKE_SOURCE_DIR}/../post/algsCom/license_plate_detection/main_all.cpp)
# add_executable(faceServer_new ${main} ${face_infer} ${post}  ${common} ${track}  ${rknn_src}) 
add_executable(carLPR ${main} ${face_infer} ${car_infer} ${post}  ${common} ${track}  ${rknn_src}   ${CMAKE_SOURCE_DIR}/prefer/get_videoffmpeg.cpp ) #qk ffmpeg
