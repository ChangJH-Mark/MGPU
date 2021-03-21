cmake_minimum_required(VERSION 3.10)
project(FASTGPU VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
aux_source_directory(src/server SERVER_SRCS)
aux_source_directory(src/common COMMON_SRCS)
add_executable(main ${COMMON_SRCS} ${SERVER_SRCS} main.cc)
add_library(client include/client/api.h src/client.cpp src/common/IPC.cpp include/common/message.h)

set(EXECUTABLE_OUTPUT_PATH /opt/custom/bin)
set(LIBRARY_OUTPUT_PATH /opt/custom/lib)

include_directories(./include /usr/local/cuda-11.0/targets/x86_64-linux/include)

add_subdirectory(src/lib)
add_subdirectory(test)

link_directories(/usr/local/cuda-11.0/targets/x86_64-linux/lib/)
target_link_libraries(main um pthread cudart)
