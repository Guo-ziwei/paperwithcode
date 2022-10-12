# set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
cmake_minimum_required(VERSION 3.9)
project(test_code)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -W -g -pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -W -fopenmp")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)

# message(${PROJECT_SOURCE_DIR})

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(mlpnp src/mlpnp.cc)
add_library(stereo src/patchmatch.cc)
target_link_libraries(stereo ${OpenCV_LIBRARIES})

add_executable(MLPnPtest test/mlpnp_test.cpp)
add_executable(patchmatch test/patchmatchtest.cpp)
add_executable(nms_test test/nms_test.cpp)

target_link_libraries(patchmatch stereo)
target_link_libraries(MLPnPtest ${OpenCV_LIBRARIES} mlpnp)
target_link_libraries(nms_test ${OpenCV_LIBRARIES})