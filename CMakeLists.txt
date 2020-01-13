cmake_minimum_required(VERSION 3.10)

project(snn)

# 安装位置
set(CMAKE_INSTALL_PREFIX "install")

# 设置OpenBLAS的头文件目录
set(OPENBLAS_INCLUDE "~/open_source_projects/OpenBLAS-openmp-install/include")
include_directories(${OPENBLAS_INCLUDE})

# 源代码
file(GLOB SOURCE_LIST "src/*.cpp")
# 头文件
file(GLOB HEADER_LIST "src/*.h")

# 添加编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3")

# 生成动态链接库
add_library(nn SHARED ${SOURCE_LIST})
# 生成静态链接库
add_library(nn-static STATIC ${SOURCE_LIST})

install(TARGETS nn LIBRARY DESTINATION "lib")
install(TARGETS nn-static ARCHIVE DESTINATION "lib")
install(FILES ${HEADER_LIST} DESTINATION "include")
