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
#define MSG_CUDA_STREAM_CREATE 0x8
#define MSG_CUDA_STREAM_SYNCHRONIZE 0x9
#define MSG_CUDA_GET_DEVICE_COUNT 0xa
#define FUNCTION_DEFINED_MASK_ 0x10000
#define MSG_MATRIX_MUL_GPU (FUNCTION_DEFINED_MASK_ | 0x1)

inline const char *get_type_msg(uint type) {
    switch (type) {
        case MSG_CUDA_MALLOC :
            return " __cuda_malloc__ ";
        case MSG_CUDA_MALLOC_HOST :
            return " __cuda_malloc_host__ ";
        case MSG_CUDA_FREE :
            return " __cuda_free__ ";
        case MSG_CUDA_FREE_HOST :
            return " __cuda_free_host__ ";
        case MSG_CUDA_MEMSET :
            return " __cuda_memset__ ";
        case MSG_CUDA_MEMCPY :
            return " __cuda_memcpy__ ";
        case MSG_CUDA_LAUNCH_KERNEL :
            return " __cuda_launch_kernel__ ";
        case MSG_CUDA_STREAM_CREATE :
            return " __cuda_stream_create__ ";
        case MSG_CUDA_STREAM_SYNCHRONIZE :
            return " __cuda_stream_synchronize__ ";
        case MSG_CUDA_GET_DEVICE_COUNT:
            return " __cuda_get_device_count__ ";
        case MSG_MATRIX_MUL_GPU:
            return " __matrix_mul_gpu__ ";
    }
}

namespace mgpu {
    typedef struct config {
        dim3 grid {1, 1, 1};
        dim3 block {1, 1, 1};
        int share_memory {0};
        int stream {0};
    } LaunchConf;

    typedef struct Matrix{
        void * data = nullptr;
        int width = 0;
        int height = 0;
    } Matrix;

    typedef uint msg_t;
    typedef int stream_t;

    typedef struct {
    public:
        msg_t type; // message type
        uint key; // pid << 16 + stream_t
    } AbMsg; // abstract message

    typedef struct CudaGetDeviceCountMsg : public AbMsg {
    } CudaGetDeviceCountMsg;

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

    typedef struct CudaStreamCreateMsg : public AbMsg {
        uint num; // streams num
    } CudaStreamCreateMsg;

    typedef struct CudaStreamSyncMsg : public AbMsg {
    } CudaStreamSyncMsg;

    typedef struct CudaLaunchKernelMsg : public AbMsg {
        LaunchConf conf;
        char ptx[128]; // ptx ptx
        char kernel[128]; // kernel symbol
        char param[1024]; // save parameters
        size_t p_size;    // param p_size
    } CudaLaunchKernelMsg;

    typedef struct MatrixMulMsg : public AbMsg {
        Matrix A;
        Matrix B;
        LaunchConf conf;
    } MatrixMulMsg;
}

namespace mgpu {
    typedef struct CudaMallocHostRet {
        void *ptr; // unified memory address
        int shmid;  // share memory id
    } CudaMallocHostRet;
}

#endif //FASTGPU_MESSAGE_H
