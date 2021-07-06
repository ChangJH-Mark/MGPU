//
// Created by root on 2021/3/16.
//

#include "client/api.h"
#include "common/IPC.h"

using namespace mgpu;

int mgpu::default_device = 0;

int mgpu::cudaGetDeviceCount() {
    auto ipc_cli = IPCClient::get_client();
    CudaGetDeviceCountMsg msg{MSG_CUDA_GET_DEVICE_COUNT, uint(pid << 16) + default_device, DEFAULT_STREAM_};
    return ipc_cli->send(&msg);
}

void mgpu::cudaSetDevice(int device) {
    default_device = device;
}

void *mgpu::cudaMalloc(size_t size) {
    auto ipc_cli = IPCClient::get_client();
    CudaMallocMsg msg{MSG_CUDA_MALLOC, uint(pid << 16) + default_device, DEFAULT_STREAM_, size};
    return ipc_cli->send(&msg);
}

void *mgpu::cudaMallocHost(size_t size) {
    auto ipc_cli = IPCClient::get_client();
    CudaMallocHostMsg msg{MSG_CUDA_MALLOC_HOST, uint(pid << 16) + default_device, DEFAULT_STREAM_, size};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaFree(void *devPtr) {
    auto ipc_cli = IPCClient::get_client();
    CudaFreeMsg msg{MSG_CUDA_FREE, uint(pid << 16) + default_device, DEFAULT_STREAM_, devPtr};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaFreeHost(void *ptr) {
    auto ipc_cli = IPCClient::get_client();
    CudaFreeHostMsg msg{MSG_CUDA_FREE_HOST, uint(pid << 16) + default_device, DEFAULT_STREAM_, ptr};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaMemset(void *devPtr, int value, size_t count) {
    auto ipc_cli = IPCClient::get_client();
    CudaMemsetMsg msg{MSG_CUDA_MEMSET, uint(pid << 16) + default_device, DEFAULT_STREAM_, devPtr, value, count};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaMemcpy(void *dst, const void * src, size_t count, ::cudaMemcpyKind kind) {
    auto ipc_cli = IPCClient::get_client();
    CudaMemcpyMsg msg{MSG_CUDA_MEMCPY, uint(pid << 16) + default_device, DEFAULT_STREAM_, dst, src, count, kind};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaStreamCreate(stream_t *stream) {
    auto ipc_cli = IPCClient::get_client();
    CudaStreamCreateMsg msg{MSG_CUDA_STREAM_CREATE, uint(pid << 16) + default_device, DEFAULT_STREAM_};
    return ipc_cli->send(&msg, stream);
}

bool mgpu::cudaStreamSynchronize(stream_t stream) {
    auto ipc_cli = IPCClient::get_client();
    CudaStreamSyncMsg msg{MSG_CUDA_STREAM_SYNCHRONIZE, uint(pid << 16) + default_device, stream};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaEventCreate(event_t *event) {
    auto ipc_cli = IPCClient::get_client();
    CudaEventCreateMsg msg{MSG_CUDA_EVENT_CREATE, uint(pid << 16) + default_device};
    return ipc_cli->send(&msg, event);
}

bool mgpu::cudaEventDestroy(event_t event) {
    auto ipc_cli = IPCClient::get_client();
    CudaEventDestroyMsg msg{MSG_CUDA_EVENT_DESTROY, uint(pid << 16) + default_device};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaEventRecord(event_t event, stream_t stream) {
    auto ipc_cli = IPCClient::get_client();
    CudaEventRecordMsg msg{MSG_CUDA_EVENT_RECORD, uint(pid << 16) + default_device, DEFAULT_STREAM_, event, stream};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaEventSynchronize(event_t event) {
    auto ipc_cli = IPCClient::get_client();
    CudaEventSyncMsg msg{MSG_CUDA_EVENT_SYNCHRONIZE, uint(pid << 16) + default_device, DEFAULT_STREAM_, event};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaEventElapsedTime(float *ms, event_t start, event_t end) {
    auto ipc_cli = IPCClient::get_client();
    CudaEventElapsedTimeMsg msg{MSG_CUDA_EVENT_ELAPSED_TIME, uint(pid << 16) + default_device, DEFAULT_STREAM_, start, end};
    return ipc_cli->send(&msg, ms);
}

std::future<void*> mgpu::matrixMul_MGPU(Matrix A, Matrix B, LaunchConf launchConf) {
    if(A.width != B.height)
    {
        perror("can't multi matrix");
        exit(1);
    }
    auto ipc_cli = IPCClient::get_client();
    MatrixMulMsg msg{MSG_MATRIX_MUL_GPU, uint(pid << 16) + default_device, DEFAULT_STREAM_, A, B, launchConf};
    return std::move(ipc_cli->send(&msg));
}