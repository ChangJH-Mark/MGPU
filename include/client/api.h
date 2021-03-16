//
// Created by root on 2021/3/15.
//

#ifndef FASTGPU_API_H
#define FASTGPU_API_H
#include <stddef.h>
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
    void *gpuMalloc(size_t size);

    // communicate with server, set @size bytes with @value at @addr on gpu.
    // return 0 on success, -1 on failure
    int gpuMemset(void *addr, int value, size_t size);

    // communicate with server, copy from @src to @dest, @size bytes
    // @stream = -1 equal default stream
    // return 0 on success, -1 on failure
    int memcpy(void *dest, void *src, size_t size, int kind, int stream = -1);

    // communicate with server, start kernel with @conf set @param
    void launchKernel(config conf, void *param);
}
#endif //FASTGPU_API_H
