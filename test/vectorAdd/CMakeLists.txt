
add_executable(vectorAddMappedMemory vectorAddMappedMemory.cu)
add_executable(vectorAdd vectorAdd.cu)

target_link_libraries(vectorAdd um)
target_link_libraries(vectorAddMappedMemory um)