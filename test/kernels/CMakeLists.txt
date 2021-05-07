add_custom_command(OUTPUT vecAdd.ptx specify_sm.ptx matrixMul.ptx
        COMMENT build vecAdd.ptx specify_sm.ptx matrixMul.ptx
        DEPENDS vecAdd.cu specify_sm.cu matrixMul.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/vecAdd.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/specify_sm.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/matrixMul.cu
        )
add_custom_target(ptx_kernels ALL
        DEPENDS vecAdd.ptx specify_sm.ptx matrixMul.ptx
        SOURCES vecAdd.cu specify_sm.cu matrixMul.cu
        )