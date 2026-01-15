cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo)

set(CMAKE_SYSTEM_NAME Linux)
set(TARGET_SOC rk3588)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")



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

# rknn api
if(TARGET_SOC STREQUAL "rk356x")
  set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/runtime/RK356X/${CMAKE_SYSTEM_NAME}/librknn_api)
elseif(TARGET_SOC STREQUAL "rk3588")
  set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
else()
  message(FATAL_ERROR "TARGET_SOC is not set, ref value: rk356x or rk3588 or rv110x")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${CMAKE_ANDROID_ARCH_ABI}/librknnrt.so)
else()
  # if (CMAKE_C_COMPILER MATCHES "aarch64")
    set(LIB_ARCH aarch64)
  # else()
  #   set(LIB_ARCH armhf)
  # endif()
  set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
endif()
include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn)
include_directories(${CMAKE_SOURCE_DIR}/infer/include)

# opencv
# if (CMAKE_SYSTEM_NAME STREQUAL "Android")
#     set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rd_rknn/opencv/OpenCV-android-sdk/sdk/native/jni/abi-${CMAKE_ANDROID_ARCH_ABI})
# else()
#   if(LIB_ARCH STREQUAL "armhf")
#     set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rd_rknn/opencv/opencv-linux-armhf/share/OpenCV)
#   else()
#     set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../3rd_rknn/opencv/opencv-linux-aarch64/share/OpenCV)
#   endif()
# endif()

include_directories(/civi/miniconda3/include)


#rga
#if(TARGET_SOC STREQUAL "rk356x")
#  set(RGA_PATH ${CMAKE_SOURCE_DIR}/3rd_rknn/rga/RK356X)
#elseif(TARGET_SOC STREQUAL "rk3588")
#  set(RGA_PATH ${CMAKE_SOURCE_DIR}/3rd_rknn/rga/RK3588)
#else()
#  message(FATAL_ERROR "TARGET_SOC is not set, ref value: rk356x or rk3588")
#endif()
#if (CMAKE_SYSTEM_NAME STREQUAL "Android")
 # set(RGA_LIB ${RGA_PATH}/lib/Android/${CMAKE_ANDROID_ARCH_ABI}/librga.so)
#else()
  # if (CMAKE_C_COMPILER MATCHES "aarch64")
#    set(LIB_ARCH aarch64)
  # else()
  #   set(LIB_ARCH armhf)
  # endif()
  #set(RGA_LIB ${RGA_PATH}/lib/Linux//${LIB_ARCH}/librga.so)
#endif()
#include_directories( ${RGA_PATH}/include)

set(CMAKE_INSTALL_RPATH "lib")

# rknn_yolov5_demo
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/infer/rknn_Mclass/include)
#include_directories( /home/blueberry/test/opencv/include)
#link_directories(/home/blueberry/test/opencv/lib)

link_directories(/media/blueberry/595d3676-4ee2-4aa9-b39d-bcd412f9f69a/miniconda3/lib/)
link_libraries(
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  )
set(rknn_src ${CMAKE_SOURCE_DIR}/infer/rknn_Mclass/src/rknn_detector.cpp ${CMAKE_SOURCE_DIR}/infer/rknn_Mclass/src/postprocess.cc)