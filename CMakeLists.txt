cmake_minimum_required(VERSION 3.18)
project(FASTGPU VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)
enable_language(CUDA)
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
    set(CMAKE_CUDA_ARCHITECTURES 50)
endif()

set(CUDA_ROOT /usr/local/cuda)
set(OUTPUT_ROOT /opt/custom)
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_ROOT}/bin)
set(LIBRARY_OUTPUT_PATH ${OUTPUT_ROOT}/lib)

link_directories(${CUDA_ROOT}/lib64)

include_directories(./include ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(src)
add_subdirectory(test)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include
        DESTINATION ${OUTPUT_ROOT})