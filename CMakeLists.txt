cmake_minimum_required(VERSION 3.10)
project(FASTGPU VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)
enable_language(CUDA)
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
    set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

link_directories(${CUDA_ROOT}/targets/x86_64-linux/lib/)
aux_source_directory(src/server SERVER_SRCS)
aux_source_directory(src/common COMMON_SRCS)
add_executable(mgpu ${COMMON_SRCS} ${SERVER_SRCS} main.cc)
add_library(client include/client/api.h src/api.cpp src/common/IPC.cpp include/common/message.h)

set(OUTPUT_ROOT /opt/custom)
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_ROOT}/bin)
set(LIBRARY_OUTPUT_PATH ${OUTPUT_ROOT}/lib)

set(CUDA_ROOT /usr/local/cuda-11.0)

include_directories(./include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(mgpu um pthread cudart cuda)

add_subdirectory(src/lib)
add_subdirectory(test)