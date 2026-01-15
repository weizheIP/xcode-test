
cmake_minimum_required(VERSION 3.10)
project(libocsort)
set(CMAKE_THREAD_LIBS_INIT "-lpthread")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE} /O2") 
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_RELEASE} /O2")

set(CMAKE_CXX_STANDARD 17)

# Linker external library
# SET(Eigen3_DIR "C:/eigen-3.4.0/build")
find_package(Eigen3 REQUIRED)
# set(OpenCV_DIR "C:/opencv")
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# set(CMAKE_PREFIX_PATH /media/ps/data1/liuym/chipsdk/torch/torch/libtorch113/share/cmake/Torch/) 
set(CMAKE_PREFIX_PATH /opt/conda/lib/python3.8/site-packages/torch/)
find_package(Torch REQUIRED)


file(GLOB SRC_LIST src/*.cpp)

include_directories(include)

add_executable(test ${SRC_LIST})
target_link_libraries(test PUBLIC Eigen3::Eigen ${OpenCV_LIBS}  ${TORCH_LIBRARIES} pthread)

# export PATH="/usr/local/cuda/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"





# macro(ncnn_add_example name)
#     add_executable(${name} ${name}.cpp)
#     if(OpenCV_FOUND)
#         target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})
#         target_link_libraries(${name} PRIVATE ncnn ${OpenCV_LIBS})
#     elseif(NCNN_SIMPLEOCV)
#         target_compile_definitions(${name} PUBLIC USE_NCNN_SIMPLEOCV)
#         target_link_libraries(${name} PRIVATE ncnn)
#     endif()

#     # add test to a virtual project group
#     set_property(TARGET ${name} PROPERTY FOLDER "examples")
# endmacro()

# if(NCNN_PIXEL)
#     find_package(OpenCV QUIET COMPONENTS opencv_world)
#     # for opencv 2.4 on ubuntu 16.04, there is no opencv_world but OpenCV_FOUND will be TRUE
#     if("${OpenCV_LIBS}" STREQUAL "")
#         set(OpenCV_FOUND FALSE)
#     endif()
#     if(NOT OpenCV_FOUND)
#         find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs videoio)
#     endif()
#     if(NOT OpenCV_FOUND)
#         find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
#     endif()

#     if(OpenCV_FOUND OR NCNN_SIMPLEOCV)
#         if(OpenCV_FOUND)
#             message(STATUS "OpenCV library: ${OpenCV_INSTALL_PATH}")
#             message(STATUS "    version: ${OpenCV_VERSION}")
#             message(STATUS "    libraries: ${OpenCV_LIBS}")
#             message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#             if(${OpenCV_VERSION_MAJOR} GREATER 3)
#                 set(CMAKE_CXX_STANDARD 11)
#             endif()
#         endif()

#         include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
#         include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)
#         include_directories(include)
#         include_directories(/usr/local/include/eigen3)

#         ncnn_add_example(squeezenet)
#         ncnn_add_example(squeezenet_c_api)
#         ncnn_add_example(fasterrcnn)
#         ncnn_add_example(rfcn)
#         ncnn_add_example(yolov2)
#         ncnn_add_example(yolov3)
#         if(OpenCV_FOUND)
#             ncnn_add_example(yolov4)
#         endif()
#         ncnn_add_example(yolov5)
#         ncnn_add_example(yolox)
#         ncnn_add_example(mobilenetv2ssdlite)
#         ncnn_add_example(mobilenetssd)
#         ncnn_add_example(squeezenetssd)
#         ncnn_add_example(shufflenetv2)
#         ncnn_add_example(peleenetssd_seg)
#         ncnn_add_example(simplepose)
#         ncnn_add_example(retinaface)
#         ncnn_add_example(yolact)
#         ncnn_add_example(nanodet)
#         ncnn_add_example(scrfd)
#         ncnn_add_example(scrfd_crowdhuman)
#         ncnn_add_example(rvm)
#         file(GLOB My_Source_Files src/*.cpp)
#         add_executable(bytetrack ${My_Source_Files})
#         if(OpenCV_FOUND)
#             target_include_directories(bytetrack PRIVATE ${OpenCV_INCLUDE_DIRS})
#             target_link_libraries(bytetrack PRIVATE ncnn ${OpenCV_LIBS})
#         elseif(NCNN_SIMPLEOCV)
#             target_compile_definitions(bytetrack PUBLIC USE_NCNN_SIMPLEOCV)
#             target_link_libraries(bytetrack PRIVATE ncnn)
#         endif()
#         # add test to a virtual project group
#         set_property(TARGET bytetrack PROPERTY FOLDER "examples")
#     else()
#         message(WARNING "OpenCV not found and NCNN_SIMPLEOCV disabled, examples won't be built")
#     endif()
# else()
#     message(WARNING "NCNN_PIXEL not enabled, examples won't be built")
# endif()
