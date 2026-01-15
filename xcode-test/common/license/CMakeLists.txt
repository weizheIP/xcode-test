
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wfatal-errors -w  -O0 -D_MWAITXINTRIN_H_INCLUDED")

set(main ${CMAKE_SOURCE_DIR}/license.cpp)
set(base64 ${CMAKE_SOURCE_DIR}/../base64/base64.cpp)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../base64/
)

link_libraries(
crypto
)


add_executable(test ${main} ${base64})