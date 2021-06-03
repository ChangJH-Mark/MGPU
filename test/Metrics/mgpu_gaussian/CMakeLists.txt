add_executable(mgpu_gaussian gaussian.cpp)
target_link_libraries(mgpu_gaussian client)

set(PTX_NAME_FLAG -o ${OUTPUT_ROOT}/ptx/gaussian.ptx)
add_custom_command(OUTPUT gaussian.ptx
        COMMENT build gaussian.ptx
        DEPENDS gaussian.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${PTX_NAME_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/gaussian.cu
        )
add_custom_target(gs.ptx ALL
        DEPENDS gaussian.ptx
        SOURCES gaussian.cu
        )

unset(PTX_NAME_FLAG)

install(TARGETS mgpu_gaussian
        DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
install(PROGRAMS run_mgpu_gaussian
        DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
