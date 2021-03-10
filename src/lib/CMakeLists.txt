add_library(um Allocator.cc UnifyPointer.cc)

include_directories("/usr/local/cuda-11.0/targets/x86_64-linux/include")

target_include_directories(um PUBLIC ${CMAKE_SOURCE_DIR}/include)