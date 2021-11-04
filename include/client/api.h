//
// Created by root on 2021/3/15.
//

#ifndef FASTGPU_API_H
#define FASTGPU_API_H
#include <stddef.h>
#include <sys/socket.h>
#include <cuda_runtime.h>
#include <future>
#include "common/message.h"
#include "common/IPC.h"
#include "common/helper.h"
#define DEFAULT_STREAM_ 0x0

namespace mgpu {
    // cudaApi
    struct config;

    extern int default_device;

    // communicate with server, get device count
    int cudaGetDeviceCount();

    // set @device used by this thread
    void cudaSetDevice(int device);

    // communicate with server, call cudaMalloc with @p_size bytes synchronously
    // return gpu memory pointer, 0 on failure
    void *cudaMalloc(size_t size);
    void *mockMalloc(size_t size);

    // communicate with server, call cudaMallocHost with @p_size bytes synchronously
    // return host memory pointer, 0 on failure.
    void *cudaMallocHost(size_t size);

    // communicate with server, call cudaFree to free memory at @ptr
    bool cudaFree(void * devPtr);

    // communicate with server, call cudaFreeHost to free page-locked memory @ptr
    bool cudaFreeHost(void * ptr);

    // communicate with server, call cudaMemset
    // initializes @count bytes @devPtr memory to @value
    bool cudaMemset(void *devPtr, int value, size_t count);

    // communicate with server, call cudaMemcpy
    // copy @count bytes from @src to @dst, use @kind distinguish cpyD2H, cpyD2D, cpyH2D, cpyH2H
    bool cudaMemcpy(void *dst, const void *src, size_t count, ::cudaMemcpyKind kind);

    // communicate with server, call cudaStreamCreate
    // create @num streams and save handler in @streams
    bool cudaStreamCreate(stream_t * stream);

    // communicate with server, call cudaStreamSynchronize on @stream
    bool cudaStreamSynchronize(stream_t stream);

    // communicate with server, call cudaEventCreate
    // create event and save handler in @event
    bool cudaEventCreate(event_t * event);
    bool cudaEventDestroy(event_t event);

    // communicate with server, call cudaEventRecord
    // record @event on @stream
    bool cudaEventRecord(event_t event, stream_t stream = nullptr);

    // communicate with server, call cudaEventSynchronize
    // synchronize on @event
    bool cudaEventSynchronize(event_t event);

    // communicate with server, call cudaEventElapsedTime
    // calculate @ms from event @start to @end
    bool cudaEventElapsedTime(float * ms, event_t start, event_t end);

    // communicate with server, start kernel with @conf set @param
    template<typename... Args>
    bool cudaLaunchKernel(LaunchConf conf, const char* name, const char* kernel, Args... args) {
        auto ipc_cli = IPCClient::get_client();
        CudaLaunchKernelMsg msg{MSG_CUDA_LAUNCH_KERNEL, uint(pid << 16) + default_device, conf.stream, conf};
        strcpy(msg.ptx, name);
        strcpy(msg.kernel, kernel);
        msg.p_size = fillParameters(msg.param, 0, args...);
        return ipc_cli->send(&msg);
    }

    template<typename... Args>
    bool mockLaunchKernel(LaunchConf conf, const char*name, const char* kernel, Args... args) {
        auto ipc_cli = IPCClient::get_client();
        MockLaunchKernelMsg msg{MSG_MOCK_LAUNCH_KERNEL, uint(pid << 16) + default_device, conf.stream, conf};
        strcpy(msg.ptx, name);
        strcpy(msg.kernel, kernel);
        msg.p_size = fillParameters(msg.param, 0, args...);
        return ipc_cli->send(&msg);
    }
}
namespace mgpu{
    // multi GPU function
    struct Matrix;
    std::future<void*> matrixMul_MGPU(Matrix A, Matrix B, LaunchConf launchConf);

    // mult task multi gpu
    MulTaskRet MulTaskMulGPU(uint nTask, Task* tasks);
}
#endif //FASTGPU_API_H
