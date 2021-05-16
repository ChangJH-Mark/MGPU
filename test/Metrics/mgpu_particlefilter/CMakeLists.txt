add_executable(mgpu_particlefilter_float ex_particle_CUDA_float_seq.cpp)
add_executable(mgpu_particlefilter_naive ex_particle_CUDA_naive_seq.cpp)

target_link_libraries(mgpu_particlefilter_float client)
target_link_libraries(mgpu_particlefilter_naive client)

set(PTX_NAME_FLAG -o ${OUTPUT_ROOT}/ptx/particle_filter.ptx)
add_custom_command(OUTPUT p_filter.ptx
        COMMENT build particle_filter.ptx p_filter_naive.ptx
        DEPENDS particle_filter.cu particle_filter_naive.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} ${PTX_NAME_FLAG} ${CMAKE_CURRENT_SOURCE_DIR}/particle_filter.cu
        COMMAND ${CMAKE_CUDA_COMPILER} ${GENERAL_PTX_FLAG} -o ${OUTPUT_ROOT}/ptx/particle_filter_naive.ptx ${CMAKE_CURRENT_SOURCE_DIR}/particle_filter_naive.cu
        )

add_custom_target(particle_filter.ptx ALL
        DEPENDS p_filter.ptx
        SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/particle_filter.cu ${CMAKE_CURRENT_SOURCE_DIR}/particle_filter_naive.cu
        )
unset(PTX_NAME_FLAG)
install(TARGETS mgpu_particlefilter_float mgpu_particlefilter_naive
        DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
install(PROGRAMS run_mgpu_particlefilter
        DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)