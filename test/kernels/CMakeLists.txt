add_custom_command(OUTPUT vectorAdd.cubin specify_sm.cubin matrixMul.cubin
        COMMENT build vectorAdd.cubin specify_sm.cubin matrixMul.cubin
        DEPENDS vectorAdd.cu specify_sm.cu matrixMul.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/vectorAdd.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/specify_sm.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/matrixMul.cu
        )
add_custom_target(ptx_kernels ALL
        DEPENDS vectorAdd.cubin specify_sm.cubin matrixMul.cubin
        SOURCES vectorAdd.cu specify_sm.cu matrixMul.cu
        )