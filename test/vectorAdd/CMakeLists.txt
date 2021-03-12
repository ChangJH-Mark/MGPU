find_package(CUDA REQUIRED)
if (${CUDA_FOUND})
include_directories("/usr/local/cuda-11.0/targets/x86_64-linux/include")
set(CUDALIBDIR "/usr/local/cuda-11.0/lib64")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; "-gencode arch=compute_50,code=sm_50")

cuda_add_executable(vectorAdd vectorAdd.cu)
cuda_add_executable(vectorAddMappedMemory vectorAddMappedMemory.cu)

target_link_libraries(vectorAdd um)
target_link_libraries(vectorAddMappedMemory um)
else()
message("fail to find cuda package")
endif()