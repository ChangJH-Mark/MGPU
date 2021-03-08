add_executable(utilization Utilization.cpp)
include_directories("/usr/local/cuda/include")
target_link_libraries(utilization /usr/lib/x86_64-linux-gnu/libnvidia-ml.so)