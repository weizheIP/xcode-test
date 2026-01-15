

# ffmpeg opencv
include_directories(
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/ffmpeg_cuda/include
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/opencv-3.4.12/include
)
link_directories(
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/opencv-3.4.12/
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/ffmpeg_cuda/lib
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/boost_1.78.0/lib
)


# cuda
# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# find_package(CUDA REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA) 
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


# tensorrt
if(X86_TENSORRT)
  include_directories(/usr/include/x86_64-linux-gnu/)
  link_directories(/usr/lib/x86_64-linux-gnu/)
  link_libraries(
    cuda cublas curand cudnn 
    cudart nvinfer nvinfer_plugin nvonnxparser
  )
endif()

# torch库
if(X86_TS)
  set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/libtorch/share/cmake/Torch)
  # set(CMAKE_PREFIX_PATH /media/ps/data1/liuym/gitlab/algsdk2024/3rd/3rd_x86/libtorch271_cu126/share/cmake/Torch)
  find_package(Torch REQUIRED)
  include_directories(
      ${CMAKE_SOURCE_DIR}/infer/torch
      # /media/ps/data1/liuym/gitlab/algsdk2024/3rd/3rd_x86/libtorch271_cu126/include
      ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/libtorch/include
  )
  link_libraries(
      ${TORCH_LIBRARIES}
  )
endif()
