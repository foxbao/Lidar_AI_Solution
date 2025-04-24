# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.12)  # 建议升级到 CMake 3.12+
project(centerpoint)

# 设置架构和路径（覆盖原有硬编码路径）
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 -Wextra -Wall -Wno-deprecated-declarations -O3")

# 1. 显式指定 TensorRT 和 CUDA 路径（与你的环境变量一致）
set(TENSORRT_ROOT "/home/baojiali/Downloads/TensorRT-8.6.1.6")  # 直接匹配你的安装路径
set(TENSORRT_INCLUDE "$ENV{TensorRT_Inc}")
set(TENSORRT_LIB "$ENV{TensorRT_Lib}")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")  # 匹配你的 CUDA 路径

# 2. 查找 CUDA 和 TensorRT 库
find_package(CUDA REQUIRED)
message(STATUS "CUDA Found: ${CUDA_VERSION}")

# 3. 包含目录
include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE}
  "3rdparty/libspconv/include"
  "include"
)

# 4. 链接目录
link_directories(
  ${TENSORRT_LIB}
  ${CUDA_LIBRARY_DIRS}
  "3rdparty/libspconv/lib/${arch}"
)

# 5. 添加可执行文件
file(GLOB_RECURSE SOURCE_FILES 
  "src/*.cu"
  "src/*.cpp"
)

cuda_add_executable(${PROJECT_NAME} 
  "main.cpp" 
  ${SOURCE_FILES}
)

# 6. 链接库（使用绝对路径避免歧义）
target_link_libraries(${PROJECT_NAME}
  "${TENSORRT_LIB}/libnvinfer.so"
  "${TENSORRT_LIB}/libnvinfer_plugin.so"  # 通常需要同时链接
  "libspconv.so"
  ${CUDA_LIBRARIES}
  stdc++fs  # 可能需要 C++17 文件系统库
)

# 7. 打印调试信息
message(STATUS "======================================")
message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE}")
message(STATUS "TensorRT Library: ${TENSORRT_LIB}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
message(STATUS "======================================")