cmake_minimum_required(VERSION 3.10)
project(FASTGPU VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
aux_source_directory(. ROOT_SRCS)
add_executable(main ${ROOT_SRCS} src/server/server.cpp src/server/Device.cpp)

set(EXECUTABLE_OUTPUT_PATH /opt/custom/bin)
set(LIBRARY_OUTPUT_PATH /opt/custom/lib)

include_directories(./include /usr/local/cuda-11.0/targets/x86_64-linux/include)

add_subdirectory(src/lib)
add_subdirectory(test)

link_directories(/usr/local/cuda-11.0/targets/x86_64-linux/lib/)
target_link_libraries(main um pthread cudart)
