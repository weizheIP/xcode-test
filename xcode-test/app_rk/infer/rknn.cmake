

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--allow-shlib-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wl,--allow-shlib-undefined")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__RKNN__")

if(BUILD_DBDS_API)# AND not xxx
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__DBDS__")
endif()

if(BUILD_TDDS_API)# AND not xxx
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__TDDS__")
endif()

set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/runtime/RK3588/${CMAKE_SYSTEM_NAME}/librknn_api)
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)
include_directories(${RKNN_API_PATH}/include)
include_directories(${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn)


include_directories(
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/opencv4/
  # /usr/include/rga/
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/mpp/RK3588/include
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/runtime/RK3588/Linux/librknn_api/include
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/ffmpeg_mpp/include

  ${CMAKE_SOURCE_DIR}/infer/rknn_Mclass/include
  ${CMAKE_SOURCE_DIR}/prefer/MPP
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/librga-main/include/
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/mpp-develop/mpp/base/inc/
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/mpp-develop/inc
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/mpp-develop/mpp/inc
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/mpp-develop/osal/inc
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/mpp-develop/utils    
  /usr/include/rockchip   
)

link_directories(
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/ffmpeg_mpp
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/lib/open4.5
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_rknn/lib
)

set(prefer ${CMAKE_SOURCE_DIR}/prefer/get_videoffmpeg.cpp)

#include_directories( ${RGA_PATH}/include)
# set(RGA_PATH ${CMAKE_SOURCE_DIR}/3rd_rknn/rga/RK3588)
# set(RGA_LIB ${RGA_PATH}/lib/Linux//${LIB_ARCH}/librga.so)

link_libraries(
  ${RKNN_RT_LIB}
  ${RGA_LIB}
  )
set(rknn_src ${CMAKE_SOURCE_DIR}/infer/rknn_Mclass/src/rknn_detector.cpp ${CMAKE_SOURCE_DIR}/infer/rknn_Mclass/src/postprocess.cc)