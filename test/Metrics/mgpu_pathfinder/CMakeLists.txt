add_executable(mgpu_pathfinder pathfinder.cpp)
target_link_libraries(mgpu_pathfinder client)

set(PTX_NAME_FLAG -o ${OUTPUT_ROOT}/ptx/pathfinder.ptx)
add_custom_command(OUTPUT pathfinder.ptx
        COMMENT build pathfinder.ptx
        DEPENDS dynproc_kernel.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${PTX_NAME_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/dynproc_kernel.cu
        )
add_custom_target(pf.ptx ALL
        DEPENDS pathfinder.ptx
        SOURCES dynproc_kernel.cu
        )

unset(PTX_NAME_FLAG)
install(TARGETS mgpu_pathfinder
        DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
install(PROGRAMS run_mgpu_pathfinder
        DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)