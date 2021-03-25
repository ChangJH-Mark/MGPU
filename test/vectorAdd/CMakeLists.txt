
add_executable(vectorAddMappedMemory vectorAddMappedMemory.cu)
add_executable(vectorAdd vectorAdd.cu)

target_link_libraries(vectorAdd um)
target_link_libraries(vectorAddMappedMemory um)

target_compile_options(vectorAdd PUBLIC --generate-code arch=compute_50,code=sm_50)
target_compile_options(vectorAddMappedMemory PUBLIC --generate-code arch=compute_50,code=sm_50)