

# ffmpeg
include_directories(
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/ffmpeg_cuda/include
  # ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/opencv-3.4.12/include
)
link_directories(
  # ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/opencv-3.4.12/
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/ffmpeg_cuda/lib
  ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/boost_1.78.0/lib
)
link_libraries(avcodec avformat avutil swresample swscale)

# opencv
find_package(OpenCV REQUIRED)
link_libraries( ${OpenCV_LIBS} )

# cuda
# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.1")
# option(WITH_NVTOOLS "Enable NVIDIA Tools Extension" OFF)
find_package(CUDA REQUIRED)
# include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA) 
# include_directories(/usr/local/cuda-12.1/include)
# link_directories(/usr/local/cuda-12.1/lib64)


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
  # set(CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/libtorch/share/cmake/Torch)
  # set(CMAKE_PREFIX_PATH /media/ps/data1/liuym/gitlab/algsdk2024/3rd/3rd_x86/libtorch271_cu126/share/cmake/Torch) # 报错
  set(CMAKE_PREFIX_PATH /usr/local/lib/python3.12/dist-packages/torch/share/cmake/Torch)  # 容器nvcr.io/nvidia/pytorch:25.05-py3
  find_package(Torch REQUIRED)
  include_directories(
      ${CMAKE_SOURCE_DIR}/infer/torch
      # /media/ps/data1/liuym/gitlab/algsdk2024/3rd/3rd_x86/libtorch271_cu126/include
      /usr/local/lib/python3.12/dist-packages/torch/include
      # ${CMAKE_SOURCE_DIR}/../3rd/3rd_x86/libtorch/include
  )
  link_libraries(
      ${TORCH_LIBRARIES}
  )
endif()


# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
# export PATH=$PATH:/usr/local/cuda/bin
# export CUDA_HOME=$CUDA_HOME:/usr/local/cuda