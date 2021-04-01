set(GENERAL_PTX_FLAG --generate-code arch=compute_50,code=sm_50 --ptx -odir=/opt/custom/ptx)
add_custom_command(OUTPUT vecAdd.ptx
        COMMENT build vecAdd.ptx
        DEPENDS vecAdd.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/vecAdd.cu
        )
add_custom_target(ptx_kernels ALL
        DEPENDS vecAdd.ptx
        SOURCES vecAdd.cu
        )