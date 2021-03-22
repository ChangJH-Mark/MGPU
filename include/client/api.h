//
// Created by root on 2021/3/15.
//

#ifndef FASTGPU_API_H
#define FASTGPU_API_H
#include <stddef.h>
#include <sys/socket.h>
#include <cuda_runtime.h>
#include "common/message.h"

namespace mgpu {

    typedef struct config {
        dim3 grid;
        dim3 block;
        int share_memory;
        int stream;
    }config;


    // communicate with server, create @size streams
    // return created stream size, -1 on failure
    int createStream(size_t size);

    // communicate with server, call cudaMalloc with @size bytes synchronously
    // return gpu memory pointer, 0 on failure
    void *cudaMalloc(size_t size);

    // communicate with server, call cudaMallocHost with @size bytes synchronously
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
    bool cudaMemcpy(void *dst, const void * src, size_t count, ::cudaMemcpyKind kind);

    // communicate with server, copy from @src to @dest, @size bytes
    // @stream = -1 equal default stream
    // return 0 on success, -1 on failure
    int memcpy(void *dest, void *src, size_t size, int kind, int stream = -1);

    // communicate with server, start kernel with @conf set @param, param is @size bytes
    void launchKernel(config conf, void *param, size_t size);
}
#endif //FASTGPU_API_H
