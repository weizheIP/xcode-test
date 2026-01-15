cmake_minimum_required(VERSION 3.14)


project (all_algs)


# 设置默认启用的模块
option(BUILD_CAR "Build car module" OFF)
option(BUILD_FACE "Build face module" OFF)

#add_subdirectory(test_main) 
#add_subdirectory (face) 
#add_subdirectory (car/app_carLPR) 

if(BUILD_CAR)
    message(STATUS "Building car module")
    #add_subdirectory(car/app_carLPR)
    add_subdirectory(car)
elseif(BUILD_FACE)
    message(STATUS "Building face module")
    add_subdirectory(face)
else()
	message(STATUS "Building general module")
	add_subdirectory(test_main) 
endif()

# add_subdirectory (test) 



