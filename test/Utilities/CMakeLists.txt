add_executable(deviceQuery deviceQuery.cu)
target_compile_options(deviceQuery PUBLIC --generate-code arch=compute_50,code=sm_50)
target_link_libraries(deviceQuery nvidia-ml)

add_executable(baseUtility baseUtility.cu)
target_compile_options(baseUtility PUBLIC --generate-code arch=compute_61,code=sm_61 --ptxas-options=-v)