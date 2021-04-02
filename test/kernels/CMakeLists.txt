set(GENERAL_PTX_FLAG --generate-code arch=compute_50,code=sm_50 --ptx -odir=/opt/custom/ptx)
add_custom_command(OUTPUT vecAdd.ptx specify_sm.ptx
        COMMENT build vecAdd.ptx specify_sm.ptx
        DEPENDS vecAdd.cu specify_sm.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/vecAdd.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/specify_sm.cu
        )
add_custom_target(ptx_kernels ALL
        DEPENDS vecAdd.ptx specify_sm.ptx
        SOURCES vecAdd.cu specify_sm.cu
        )