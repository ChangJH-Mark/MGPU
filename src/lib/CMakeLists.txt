add_library(allocator Allocator.cc)

add_library(unify_pointer UnifyPointer.cc)

include_directories("/usr/local/cuda-11.0/targets/x86_64-linux/include")

target_include_directories(allocator PUBLIC ${CMAKE_SOURCE_DIR}/include)
target_include_directories(unify_pointer PUBLIC ${CMAKE_SOURCE_DIR}/include)