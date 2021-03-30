//
// Created by root on 2021/3/15.
//

#ifndef FASTGPU_API_H
#define FASTGPU_API_H
#include <stddef.h>
#include <sys/socket.h>
#include <cuda_runtime.h>
#include "common/message.h"
#include "common/IPC.h"

namespace mgpu {

    struct config;
    // communicate with server, create @p_size streams
    // return created stream p_size, -1 on failure
    int createStream(size_t size);

    // communicate with server, call cudaMalloc with @p_size bytes synchronously
    // return gpu memory pointer, 0 on failure
    void *cudaMalloc(size_t size);

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
    bool cudaStreamCreate(stream_t * streams, uint num);

    // communicate with server, call cudaStreamSynchronize on @stream
    bool cudaStreamSynchronize(stream_t stream);

    // communicate with server, start kernel with @conf set @param
    template<typename T, typename... Args>
    static unsigned fillParameters(char *buff, unsigned offset, T value, Args... args);

    template<typename T>
    static unsigned fillParameters(char *buff, unsigned offset, T value);

    template<typename... Args>
    bool cudaLaunchKernel(config conf, const char* name, Args... args) {
        auto ipc_cli = IPCClient::get_client();
        CudaLaunchKernelMsg msg{MSG_CUDA_LAUNCH_KERNEL, uint(pid << 16) + conf.stream, conf};
        strcpy(msg.name, name);
        msg.p_size = fillParameters(msg.param, 0, args...);
        return ipc_cli->send(&msg);
    }

    template<typename T, typename... Args>
    static unsigned fillParameters(char *buff, unsigned offset, T value, Args... args)
    {
        offset = (offset + __alignof(value) -1) & ~(__alignof(value) -1);
        memcpy((buff + offset), &value, sizeof(value));
        offset += sizeof(value);
        return fillParameters(buff, offset, args...);
    }

    template<typename T>
    static unsigned fillParameters(char *buff, unsigned int offset, T value) {
        offset = (offset + __alignof(value) -1) & ~(__alignof(value) -1);
        memcpy((buff + offset), &value, sizeof(value));
        offset += sizeof(value);
        return offset;
    }
}
#endif //FASTGPU_API_H
