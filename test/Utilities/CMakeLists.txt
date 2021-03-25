add_executable(deviceQuery deviceQuery.cu)
target_compile_options(deviceQuery PUBLIC --generate-code arch=compute_50,code=sm_50)

add_executable(concurrentKernel concurrentKernel.cu)
target_compile_options(concurrentKernel PUBLIC --generate-code arch=compute_50,code=sm_50)

add_executable(differentKernels DifferentKernels.cu)
target_compile_options(differentKernels PUBLIC --generate-code arch=compute_50,code=sm_50)

add_executable(simpleStream simpleStream.cu)
target_compile_options(simpleStream PUBLIC --generate-code arch=compute_50,code=sm_50)

add_executable(socket socket.cpp socket.cpp)