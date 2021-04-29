add_executable(mgpu_pathfinder pathfinder.cpp)
add_library(pathfinder.ptx OBJECT dynproc_kernel.cu)
target_compile_options(pathfinder.ptx PUBLIC --ptx)

target_link_libraries(mgpu_pathfinder client cudart)

install(TARGETS mgpu_pathfinder
        DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)
install(TARGETS pathfinder.ptx
        DESTINATION /opt/custom/ptx)
install(PROGRAMS run_mgpu_pathfinder
        DESTINATION ${EXECUTABLE_OUTPUT_PATH}/test)