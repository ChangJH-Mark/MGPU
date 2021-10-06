//
// Created by root on 2021/3/16.
//

#ifndef FASTGPU_MESSAGE_H
#define FASTGPU_MESSAGE_H

#include <cuda_runtime.h>
#include <cuda.h>

#define MSG_CUDA_MALLOC    0x1
#define MSG_CUDA_MALLOC_HOST 0x2
#define MSG_CUDA_FREE 0x3
#define MSG_CUDA_FREE_HOST 0x4
#define MSG_CUDA_MEMSET 0x5
#define MSG_CUDA_MEMCPY 0x6
#define MSG_CUDA_LAUNCH_KERNEL 0x7
#define MSG_MOCK_LAUNCH_KERNEL 0x10007
#define MSG_CUDA_STREAM_CREATE 0x8
#define MSG_CUDA_STREAM_SYNCHRONIZE 0x9
#define MSG_CUDA_GET_DEVICE_COUNT 0xa
#define MSG_CUDA_EVENT_CREATE 0xb
#define MSG_CUDA_EVENT_DESTROY 0xc
#define MSG_CUDA_EVENT_RECORD 0xd
#define MSG_CUDA_EVENT_SYNCHRONIZE 0xe
#define MSG_CUDA_EVENT_ELAPSED_TIME 0xf
#define FUNCTION_DEFINED_MASK_ 0x10000
#define MSG_MATRIX_MUL_GPU (FUNCTION_DEFINED_MASK_ | 0x1)
#define MSG_MUL_TASK (FUNCTION_DEFINED_MASK_ | 0x2)

#define MAX_HOST_ADDR_NUM 4
#define MAX_DEVICE_ADDR_NUM 4
#define MAX_TASK_NUM 4

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
        case MSG_MOCK_LAUNCH_KERNEL :
            return " __mock_launch_kernel__ ";
        case MSG_CUDA_STREAM_CREATE :
            return " __cuda_stream_create__ ";
        case MSG_CUDA_STREAM_SYNCHRONIZE :
            return " __cuda_stream_synchronize__ ";
        case MSG_CUDA_GET_DEVICE_COUNT:
            return " __cuda_get_device_count__ ";
        case MSG_CUDA_EVENT_CREATE:
            return " __cuda_event_create__ ";
        case MSG_CUDA_EVENT_DESTROY:
            return " __cuda_event_destroy__ ";
        case MSG_CUDA_EVENT_RECORD:
            return " __cuda_event_record__ ";
        case MSG_CUDA_EVENT_SYNCHRONIZE:
            return " __cuda_event_synchronize__ ";
        case MSG_CUDA_EVENT_ELAPSED_TIME:
            return " __cuda_event_elapsed_time__ ";
        case MSG_MATRIX_MUL_GPU:
            return " __matrix_mul_gpu__ ";
        case MSG_MUL_TASK:
            return " __mul_task__ ";
    }
}

namespace mgpu {
    typedef uint api_t;
    typedef cudaStream_t stream_t;
    typedef cudaEvent_t event_t;

    typedef struct config {
        dim3 grid{1, 1, 1};
        dim3 block{1, 1, 1};
        int share_memory{0};
        stream_t stream{nullptr};
    } LaunchConf;

    typedef struct Matrix {
        void *data = nullptr;
        int width = 0;
        int height = 0;
    } Matrix;

    typedef struct Task {
        uint hdn; // data number in host
        uint dn;  // extra device memory num needed
        size_t hds[MAX_HOST_ADDR_NUM]; // data size in host
        size_t dev_alloc_size[MAX_DEVICE_ADDR_NUM]; // extra device memory size
        LaunchConf conf;
        char ptx[128]{}; // ptx ptx
        char kernel[128]{}; // kernel symbol
        char param[1024]{}; // pre-filled with host data address and extra device mem empty addr
        size_t p_size{};    // param size
    } Task;

    typedef struct AbMsg {
    public:
        api_t type; // api type
        uint key; // pid << 16 + device
        stream_t stream; // stream
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
    } CudaStreamCreateMsg;

    typedef struct CudaStreamSyncMsg : public AbMsg {
    } CudaStreamSyncMsg;

    typedef struct CudaEventCreateMsg : public AbMsg {
    } CudaEventCreateMsg;

    typedef struct CudaEventDestroyMsg : public AbMsg {
        event_t event;
    } CudaEventDestroyMsg;

    typedef struct CudaEventRecordMsg : public AbMsg {
        event_t event;
        stream_t stream;
    } CudaEventRecordMsg;

    typedef struct CudaEventSyncMsg : public AbMsg {
        event_t event;
    } CudaEventSyncMsg;

    typedef struct CudaEventElapsedTimeMsg : public AbMsg {
        event_t start;
        event_t end;
    } CudaEventElapsedTimeMsg;

    typedef struct CudaLaunchKernelMsg : public AbMsg {
        LaunchConf conf;
        char ptx[128]{}; // ptx ptx
        char kernel[128]{}; // kernel symbol
        char param[1024]{}; // save parameters
        size_t p_size{};    // param p_size
    } CudaLaunchKernelMsg;

    // mock msg have the same member as cuda
    typedef struct CudaLaunchKernelMsg MockLaunchKernelMsg;

    typedef struct MulTaskMsg : public AbMsg {
        Task task[MAX_TASK_NUM];
        uint task_num;
    } MulTaskMsg;

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

    typedef struct MulTaskRet {
        int success;
        char msg[1024]; // error message
    } MulTaskRet;
}

#endif //FASTGPU_MESSAGE_H
