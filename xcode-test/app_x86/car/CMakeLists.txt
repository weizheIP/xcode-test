cmake_minimum_required(VERSION 3.14)

project(car)


add_definitions(-std=c++14)
add_definitions(-DAPI_EXPORTS)

#必选开关，编译http库的时候，选用poll模型，启用录像会把select(跨操作系统平台)最大数耗尽,window不支持poll
add_definitions(-DCPPHTTPLIB_USE_POLL)
add_definitions(-DCROW_MAIN)

#功能设置
# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(X86_TS ON)
add_definitions(-DX86_TS)
add_definitions(-DUSE_FACE)
add_definitions(-DUSE_LANDMARK) # 跨车位用关键点
# add_definitions(-DSTRANGERDB)
add_definitions(-DDECNOTRESIZE) #吸烟等要原图 人脸要，但是后处理那里还没写
add_definitions(-DZHIWEI_GONGKONG)

if(CMAKE_BUILD_TYPE MATCHES Debug)
add_definitions(-DDEED_DEBUG) # 调试(不用lic)，发表用release
message(STATUS "Debug...")
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wfatal-errors -w  -O0 -D_MWAITXINTRIN_H_INCLUDED")# -fsanitize=address -fno-omit-frame-pointer")

include(${CMAKE_SOURCE_DIR}/infer/tensorrt.cmake)



include_directories(
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/onnxruntime-linux-x64-gpu-1.12.1/include
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/faiss-1.7.4/
  ${CMAKE_SOURCE_DIR}/infer/face/
  ${CMAKE_SOURCE_DIR}/infer/car_plate/
)

link_directories(
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/onnxruntime-linux-x64-gpu-1.12.1/lib
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/onnx
)



set(face_infer
${CMAKE_SOURCE_DIR}/infer/face/face_det.cpp
${CMAKE_SOURCE_DIR}/infer/face/face_rec.cpp
)

set(car_infer
${CMAKE_SOURCE_DIR}/infer/car_plate/CrnnNet.cpp
${CMAKE_SOURCE_DIR}/infer/car_plate/car_det.cpp
)


include(${CMAKE_SOURCE_DIR}/../common/common.cmake)

# find_package(OpenCV 3.4.12 REQUIRED)
link_libraries(stdc++ pthread opencv_imgproc  opencv_core opencv_highgui opencv_imgcodecs opencv_video opencv_videoio opencv_dnn opencv_calib3d
avcodec avformat avutil swresample swscale
boost_thread crypt crypto ssl
faiss
onnxruntime 
)

# # 旧
# set(prefer ${CMAKE_SOURCE_DIR}/prefer/get_videoffmpeg.cpp)
# file(GLOB main ${CMAKE_SOURCE_DIR}/../post/face/*.cpp)
# add_executable(faceServer ${main} ${muxer} ${prefer} ${face_infer} ${face_infer2} ${post} ${post2} ${common} ${track} ${face1} ${face2} ${face3}${face4})

# 新
file(GLOB main ${CMAKE_SOURCE_DIR}/../post/algsCom/license_plate_detection/main_all.cpp)
set(post ${CMAKE_SOURCE_DIR}/../post/face/face_db.cpp  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp )
add_executable(car_LPR ${main} ${face_infer} ${car_infer} ${post}  ${common} ${track} ${CMAKE_SOURCE_DIR}/prefer/get_videoffmpeg.cpp)

# # 表情
# file(GLOB main ${CMAKE_SOURCE_DIR}/../post/algsCom/face_recognition/face_expression/*.cpp)
# set(post ${CMAKE_SOURCE_DIR}/../post/face/face_db.cpp  ${CMAKE_SOURCE_DIR}/../common/ConvertImage.cpp )
# add_executable(faceServer_fer ${main} ${face_infer} ${post}  ${common} ${track} )

