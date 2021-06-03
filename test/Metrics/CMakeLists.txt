if(${BUILD_TEST})
    # raw cuda program
    add_executable(pathfinder pathfinder/pathfinder.cu)
    target_link_libraries(pathfinder cudart)

    add_executable(hotspot hotspot/hotspot.cu)
    target_link_libraries(hotspot cudart)

    add_executable(sc_gpu streamcluster/streamcluster_header.cu streamcluster/streamcluster_cuda.cu streamcluster/streamcluster_cuda_cpu.cpp)
    target_link_libraries(sc_gpu cuda)

    add_executable(gaussian gaussian/gaussian.cu)
    target_link_libraries(gaussian cudart)

    # particlefilter
    add_executable(particlefilter_float particlefilter/ex_particle_CUDA_float_seq.cu)
    add_executable(particlefilter_naive particlefilter/ex_particle_CUDA_naive_seq.cu)

    target_link_libraries(particlefilter_float m cuda)
    target_link_libraries(particlefilter_naive m cuda)

    target_compile_options(particlefilter_float BEFORE
            PRIVATE -g -O3 -use_fast_math -arch sm_50)
    target_compile_options(particlefilter_naive BEFORE
            PRIVATE -g -O3 -use_fast_math -arch sm_50)

    # install executable, scripts, test data
    # executable
    install(TARGETS pathfinder hotspot sc_gpu particlefilter_float gaussian
            DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
    # run scripts
    install(PROGRAMS pathfinder/run_pathfinder hotspot/run_hotspot streamcluster/run_scgpu particlefilter/run_particlefilter gaussian/run_gaussian
            DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
    # test data
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/data
            DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
endif()

# mgpu version
add_subdirectory(mgpu_hotspot)
add_subdirectory(mgpu_pathfinder)
add_subdirectory(mgpu_streamcluster)
add_subdirectory(mgpu_particlefilter)
add_subdirectory(mgpu_gaussian)