//
// Created by root on 2021/3/16.
//

#ifndef FASTGPU_MESSAGE_H
#define FASTGPU_MESSAGE_H

#include <cuda_runtime.h>

#define MSG_CUDA_MALLOC    0x1
#define MSG_CUDA_MALLOC_HOST 0x2
#define MSG_CUDA_FREE 0x3
#define MSG_CUDA_FREE_HOST 0x4
#define MSG_CUDA_MEMSET 0x5
#define MSG_CUDA_MEMCPY 0x6
#define MSG_CUDA_LAUNCH_KERNEL 0x7

namespace mgpu {
    typedef struct config {
        dim3 grid;
        dim3 block;
        int share_memory;
        int stream;
    } config;

    typedef uint msg_t;

    typedef struct {
    public:
        msg_t type; // message type
        uint key; // pid << 16 + stream_t
    } AbMsg; // abstract message

    typedef struct CudaMallocMsg : public AbMsg {
        size_t size; // gpu memory bytes
    } CudaMallocMsg;

    typedef struct CudaMallocHostMsg : public AbMsg {
        size_t size; // host memory bytes
    } CudaMallocHostMsg;

    typedef struct CudaFreeMsg : public AbMsg {
        void *devPtr; // gpu memory pointer
    } CudaFreeMsg;

    typedef struct CudaFreeHostMsg : public AbMsg {
        void *ptr; // page-locked memory pointer
    } CudaFreeHostMsg;

    typedef struct CudaMemsetMsg : public AbMsg {
        void *devPtr;
        int value;
        size_t count;
    } CudaMemsetMsg;

    typedef struct CudaMemcpyMsg : public AbMsg {
        void *dst;
        const void *src;
        size_t count;
        cudaMemcpyKind kind;
    } CudaMemcpyMsg;

    typedef struct CudaLaunchKernelMsg : public AbMsg {
        config conf;
        char name[128]; // ptx name
        size_t p_size;    // param p_size
        char param[1024]; // save parameters
    } CudaLaunchKernelMsg;
}

namespace mgpu {
    typedef struct CudaMallocHostRet {
        void *ptr; // unified memory address
        int shmid;  // share memory id
    } CudaMallocHostRet;
}

#endif //FASTGPU_MESSAGE_H
