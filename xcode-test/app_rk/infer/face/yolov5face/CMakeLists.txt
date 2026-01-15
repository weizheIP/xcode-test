# cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo)

set(CMAKE_SYSTEM_NAME Linux)
set(TARGET_SOC rk3588)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g ")#-Wl,--allow-shlib-undefined
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 ")#-Wl,--allow-shlib-undefined



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__RKNN__")


if(BUILD_DBDS_API)# AND not xxx
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__DBDS__")
endif()

if(BUILD_TDDS_API)# AND not xxx
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__TDDS__")
endif()

# message(${BUILD_DBDS_API})
# message(${BUILD_TDDS_API})

#set opencv
#set(OpenCV_LIBS ${CMAKE_SOURCE_DIR}/opencv/lib)
#include_directories(${CMAKE_SOURCE_DIR}/opencv/include)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo_${CMAKE_SYSTEM_NAME})

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


# set(RKNN_API_PATH /userdata/rknn_Mclass/rknpu2/rknpu2-master/runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
# set(RKNN_RT_LIB ${RKNN_API_PATH}/aarch64/librknnrt.so
# ${RKNN_API_PATH}/aarch64/librknn_api.so
# )


#rga
include_directories(/usr/include/rga)

set(CMAKE_INSTALL_RPATH "lib")

# rknn_yolov5_demo
include_directories( 
    /userdata/civi2/bug/3rd_rknn/runtime/RK3588/Linux/librknn_api/include
    /usr/include/opencv4/
)

link_directories(
    /userdata/civi2/bug/infer/new_detect/lib
  /userdata/civi2/mpp-develop/build2/utils/
  /userdata/civi2/all_algs/3rd_rknn/ffmpeg_mpp
)

link_libraries(
  ${RGA_LIB}
  rknn_api
  avcodec
  avformat
  avutil
  swresample
  swscale

  opencv_core
  opencv_highgui
  opencv_imgcodecs
  opencv_imgproc
  opencv_videoio
  opencv_dnn
  opencv_calib3d

  rga
  )
add_executable(main 
${CMAKE_SOURCE_DIR}/face_detect.cpp
${CMAKE_SOURCE_DIR}/main.cpp
${CMAKE_SOURCE_DIR}/face_landmark.cpp
${CMAKE_SOURCE_DIR}/faceYolov5.cpp
${CMAKE_SOURCE_DIR}/face_recognition.cpp
${CMAKE_SOURCE_DIR}/face_eye_close.cpp
${CMAKE_SOURCE_DIR}/face_post.cpp
)