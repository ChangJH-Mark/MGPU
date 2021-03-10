find_package(CUDA REQUIRED)
if (${CUDA_FOUND})
include_directories("/usr/local/cuda-11.0/targets/x86_64-linux/include")
set(CUDALIBDIR "/usr/local/cuda-11.0/lib64")

cuda_add_executable(vectorAdd vectorAdd.cu)
cuda_add_executable(vectorAddMappedMemory vectorAddMappedMemory.cu)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; "-gencoe arch=compute_50,code=sm_50")

target_link_libraries(vectorAdd um)
target_link_libraries(vectorAddMappedMemory um)
else()
message("fail to find cuda package")
endif()