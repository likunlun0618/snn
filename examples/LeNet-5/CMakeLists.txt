cmake_minimum_required(VERSION 3.10)

project(lenet-5)

# 安装位置
set(CMake_INSTALL_PREFIX "install")

# 设置OpenBLAS的路径
set(OPENBLAS_LIB "/opt/OpenBLAS/lib")

# 设置opencv的路径
set(OPENCV_INCLUDE "/usr/local/include/opencv4")
set(OPENCV_LIB "/usr/local/lib")

# 设置snn的路径
set(SNN_INCLUDE "../../build/install/include")
set(SNN_LIB "../../build/install/lib")

# 头文件目录
include_directories(${OPENCV_INCLUDE})
include_directories(${SNN_INCLUDE})

# 库目录
link_directories(${OPENBLAS_LIB})
link_directories(${OPENCV_LIB})
link_directories(${SNN_LIB})

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3")

add_executable(lenet-5 lenet-5.cpp)
target_link_libraries(lenet-5 libnn-static.a)
target_link_libraries(lenet-5 libopenblas.a)
target_link_libraries(lenet-5 opencv_core)
target_link_libraries(lenet-5 opencv_imgcodecs)
target_link_libraries(lenet-5 opencv_imgproc)
