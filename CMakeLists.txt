cmake_minimum_required(VERSION 2.8)

project (ai_driver_rhf)

if (NOT CMAKE_BUILD_TYPE) 
    set (CMAKE_BUILD_TYPE "Release") 
endif ()

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

if (NOT MSVC)
	set(CMAKE_CXX "${CMAKE_CXX_FLAGS} -std=c++11")
endif ()

find_package(Arnold REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenImageIO REQUIRED)

# Requirements from OpenImageIO
find_package(Boost REQUIRED)
find_package(IlmBase REQUIRED)

include_directories(${ARNOLD_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIR} ${ILMBASE_INCLUDE_DIR})
include_directories(${OPENIMAGEIO_INCLUDES})
include_directories(${TBB_INCLUDE_DIRS})

link_directories(${Boost_LIBRARY_DIRS})

set(sources driver.cpp rhf_filter.cpp rhf_filter.hpp)

add_library(ai_driver_rhf SHARED ${sources})
target_link_libraries(ai_driver_rhf ${ARNOLD_LIBRARIES})
target_link_libraries(ai_driver_rhf ${OPENIMAGEIO_LIBRARY})
target_link_libraries(ai_driver_rhf ${TBB_LIBRARIES})