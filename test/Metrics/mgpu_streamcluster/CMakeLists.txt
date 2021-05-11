add_executable(mgpu_sc_gpu streamcluster_cuda.cpp streamcluster_cuda_cpu.cpp streamcluster_header.cu)
target_link_libraries(mgpu_sc_gpu client)

set(PTX_NAME_FLAG -o ${OUTPUT_ROOT}/ptx/streamcluster.ptx)
add_custom_command(OUTPUT kernel_compute_cost.ptx
        COMMENT build kernel_compute_cost.ptx
        DEPENDS kernel_compute_cost.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${PTX_NAME_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/kernel_compute_cost.cu
        )
add_custom_target(kcc.ptx ALL
        DEPENDS kernel_compute_cost.ptx
        SOURCES kernel_compute_cost.cu
        )
unset(PTX_NAME_FLAG)
install(TARGETS mgpu_sc_gpu
        DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
install(PROGRAMS run_mgpu_scgpu
        DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)