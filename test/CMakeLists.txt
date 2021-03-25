set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; "-gencode arch=compute_50,code=sm_50")
add_subdirectory(client)
add_subdirectory(vectorAdd)
add_subdirectory(Utilities)