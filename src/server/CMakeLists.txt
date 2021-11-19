set(SERVER_SRCS server.cpp Scheduler.cpp Receiver.cpp Device.cpp KernelMgr.cpp Conductor.cpp ProxyWorker.cpp main.cc)

set(CMAKE_VERBOSE_MAKEFILE on)
add_custom_command(OUTPUT exec
        COMMAND nvcc --default-stream per-thread -I${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/Conductor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/server.cpp  ${CMAKE_CURRENT_SOURCE_DIR}/Scheduler.cpp ${CMAKE_CURRENT_SOURCE_DIR}/Receiver.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/Device.cpp ${CMAKE_CURRENT_SOURCE_DIR}/KernelMgr.cpp ${CMAKE_CURRENT_SOURCE_DIR}/ProxyWorker.cpp
        ${PROJECT_SOURCE_DIR}/src/common/ThreadPool.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cc -o /opt/custom/bin/mgpu -L/usr/local/cuda/lib64 -L/opt/custom/lib
        -lrt -lpthread -lcudart -lcuda -lcump
        )

add_custom_target(mgpu ALL
        DEPENDS exec
        )