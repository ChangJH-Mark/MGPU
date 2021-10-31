set(SERVER_SRCS server.cpp Scheduler.cpp Receiver.cpp Device.cpp KernelMgr.cpp Conductor.cpp ProxyWorker.cpp main.cc)

add_executable(mgpu ${COMMON_SRCS} ${SERVER_SRCS})
target_link_libraries(mgpu um rt pthread cudart cuda cump)