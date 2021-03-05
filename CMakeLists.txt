cmake_minimum_required(VERSION 3.10)
project(FASTGPU VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
aux_source_directory(. ROOT_SRCS)
add_executable(main ${ROOT_SRCS})

set(EXECUTABLE_OUTPUT_PATH /opt/custom/bin)
set(LIBRARY_OUTPUT_DIRECTORY /opt/custom/lib)

include_directories(./include)

add_subdirectory(src/lib)
add_subdirectory(test)

target_link_libraries(main allocator unify_pointer)
