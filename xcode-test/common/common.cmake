# 通用
# boost的websocket等 /common/boost
# 跨平台的业务逻辑 /post

set(common 
${CMAKE_SOURCE_DIR}/../common/json/jsoncpp.cpp 
${CMAKE_SOURCE_DIR}/../common/mp4Rec/mp4Rec.cpp 
${CMAKE_SOURCE_DIR}/../common/base64/base64.cpp
${CMAKE_SOURCE_DIR}/../common/license/license.cpp
${CMAKE_SOURCE_DIR}/../common/http/httpprocess.cpp
${CMAKE_SOURCE_DIR}/../common/alarmVideo/alarm_video.cpp
)

aux_source_directory(${CMAKE_SOURCE_DIR}/../common/Track/src track)

include_directories(
    ${CMAKE_SOURCE_DIR}/../3rd/boost/include
    ${CMAKE_SOURCE_DIR}/../common/websocket
    ${CMAKE_SOURCE_DIR}/../common/json
    ${CMAKE_SOURCE_DIR}/../common/base64
    ${CMAKE_SOURCE_DIR}/../common/license
    ${CMAKE_SOURCE_DIR}/../common/http
    ${CMAKE_SOURCE_DIR}/../common/mp4Rec
    ${CMAKE_SOURCE_DIR}/../alarmVideo
    ${CMAKE_SOURCE_DIR}/../common
    ${CMAKE_SOURCE_DIR}/../
    ${CMAKE_SOURCE_DIR}/../common/Track/inc
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/../post
    ${CMAKE_CURRENT_SOURCE_DIR}
)

link_libraries(stdc++ pthread 
crypt
crypto
ssl
)




# 视觉：opencv和ffmpeg
# 跟踪

# link_libraries(stdc++ pthread opencv_imgproc  opencv_core opencv_highgui opencv_imgcodecs opencv_video opencv_videoio 
# opencv_dnn
# opencv_calib3d
# avcodec
# avformat
# avutil
# swresample
# swscale
# boost_thread
# crypt
# crypto
# ssl
# )