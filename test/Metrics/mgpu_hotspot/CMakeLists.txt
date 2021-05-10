add_executable(mgpu_hotspot hotspot.cpp)
target_link_libraries(mgpu_hotspot client)

set(PTX_NAME_FLAG -o ${OUTPUT_ROOT}/ptx/hotspot.ptx)
add_custom_command(OUTPUT hotspot.ptx
        COMMENT build hotspot.ptx
        DEPENDS hotspot.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${PTX_NAME_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/hotspot.cu
        )
add_custom_target(hs.ptx ALL
        DEPENDS hotspot.ptx
        SOURCES hotspot.cu
        )

unset(PTX_NAME_FLAG)

install(TARGETS mgpu_hotspot
        DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
install(PROGRAMS run_mgpu_hotspot
        DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
