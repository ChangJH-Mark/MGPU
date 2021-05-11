if(${BUILD_TEST})
    # raw cuda program
    add_executable(pathfinder pathfinder/pathfinder.cu)
    target_link_libraries(pathfinder cudart)

    add_executable(hotspot hotspot/hotspot.cu)
    target_link_libraries(hotspot cudart)

    add_executable(sc_gpu streamcluster/streamcluster_header.cu streamcluster/streamcluster_cuda.cu streamcluster/streamcluster_cuda_cpu.cpp)
    target_link_libraries(sc_gpu cuda)

    # executable
    install(TARGETS pathfinder hotspot sc_gpu
            DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
    # run scripts
    install(PROGRAMS pathfinder/run_pathfinder hotspot/run_hotspot streamcluster/run_scgpu
            DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
    # test data
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/data
            DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
endif()

# mgpu version
add_subdirectory(mgpu_hotspot)
add_subdirectory(mgpu_pathfinder)
add_subdirectory(mgpu_streamcluster)