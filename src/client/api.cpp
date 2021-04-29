//
// Created by root on 2021/3/16.
//

#include "client/api.h"
#include "common/IPC.h"

using namespace mgpu;

int mgpu::default_device = 0;

int mgpu::cudaGetDeviceCount() {
    auto ipc_cli = IPCClient::get_client();
    CudaGetDeviceCountMsg msg{MSG_CUDA_GET_DEVICE_COUNT, uint(pid << 16) + (default_device << 8) + DEFAULT_STREAM_};
    return ipc_cli->send(&msg);
}

void mgpu::cudaSetDevice(int device) {
    default_device = device;
}

void *mgpu::cudaMalloc(size_t size) {
    auto ipc_cli = IPCClient::get_client();
    CudaMallocMsg msg{MSG_CUDA_MALLOC, uint(pid << 16) + (default_device << 8) + DEFAULT_STREAM_, size};
    return ipc_cli->send(&msg);
}

void *mgpu::cudaMallocHost(size_t size) {
    auto ipc_cli = IPCClient::get_client();
    CudaMallocHostMsg msg{MSG_CUDA_MALLOC_HOST, uint(pid << 16) + (default_device << 8) + DEFAULT_STREAM_, size};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaFree(void *devPtr) {
    auto ipc_cli = IPCClient::get_client();
    CudaFreeMsg msg{MSG_CUDA_FREE, uint(pid << 16) + (default_device << 8) + DEFAULT_STREAM_, devPtr};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaFreeHost(void *ptr) {
    auto ipc_cli = IPCClient::get_client();
    CudaFreeHostMsg msg{MSG_CUDA_FREE_HOST, uint(pid << 16) + (default_device << 8) + DEFAULT_STREAM_, ptr};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaMemset(void *devPtr, int value, size_t count) {
    auto ipc_cli = IPCClient::get_client();
    CudaMemsetMsg msg{MSG_CUDA_MEMSET, uint(pid << 16) + (default_device << 8) + DEFAULT_STREAM_, devPtr, value, count};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaMemcpy(void *dst, const void * src, size_t count, ::cudaMemcpyKind kind) {
    auto ipc_cli = IPCClient::get_client();
    CudaMemcpyMsg msg{MSG_CUDA_MEMCPY, uint(pid << 16) + (default_device << 8) + DEFAULT_STREAM_, dst, src, count, kind};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaStreamCreate(stream_t *streams, uint num) {
    auto ipc_cli = IPCClient::get_client();
    CudaStreamCreateMsg msg{MSG_CUDA_STREAM_CREATE, uint(pid << 16) + (default_device << 8) + DEFAULT_STREAM_, num};
    return ipc_cli->send(&msg, streams);
}

bool mgpu::cudaStreamSynchronize(stream_t stream) {
    auto ipc_cli = IPCClient::get_client();
    CudaStreamSyncMsg msg{MSG_CUDA_STREAM_SYNCHRONIZE, uint(pid << 16) + (default_device << 8) + stream};
    return ipc_cli->send(&msg);
}

std::future<void*> mgpu::matrixMul_MGPU(Matrix A, Matrix B, LaunchConf launchConf) {
    if(A.width != B.height)
    {
        perror("can't multi matrix");
        exit(1);
    }
    auto ipc_cli = IPCClient::get_client();
    MatrixMulMsg msg{MSG_MATRIX_MUL_GPU, uint(pid << 16) + (default_device << 8) + DEFAULT_STREAM_, A, B, launchConf};
    return std::move(ipc_cli->send(&msg));
}