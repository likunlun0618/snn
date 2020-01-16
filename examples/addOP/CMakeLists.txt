cmake_minimum_required(VERSION 3.10)

project(pow)

# 安装位置
set(CMake_INSTALL_PREFIX "install")

# 设置OpenBLAS的路径
set(OPENBLAS_LIB "/opt/OpenBLAS/lib")

# 设置snn的路径
set(SNN_INCLUDE "../../build/install/include")
set(SNN_LIB "../../build/install/lib")

# 头文件目录
include_directories(${SNN_INCLUDE})

# 库目录
link_directories(${OPENBLAS_LIB})
link_directories(${SNN_LIB})

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3")

add_executable(pow main.cpp pow.cpp)
target_link_libraries(pow libnn-static.a)
target_link_libraries(pow libopenblas.a)
