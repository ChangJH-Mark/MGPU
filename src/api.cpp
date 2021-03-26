//
// Created by root on 2021/3/16.
//

#include "client/api.h"
#include "common/IPC.h"

using namespace mgpu;

void *mgpu::cudaMalloc(size_t size) {
    auto ipc_cli = IPCClient::get_client();
    CudaMallocMsg msg{MSG_CUDA_MALLOC, uint(pid << 16) + 0xffff/*default stream*/, size};
    return ipc_cli->send(&msg);
}

void *mgpu::cudaMallocHost(size_t size) {
    auto ipc_cli = IPCClient::get_client();
    CudaMallocHostMsg msg{MSG_CUDA_MALLOC_HOST, uint(pid << 16) + 0xffff, size};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaFree(void *devPtr) {
    auto ipc_cli = IPCClient::get_client();
    CudaFreeMsg msg{MSG_CUDA_FREE, uint(pid << 16) + 0xffff, devPtr};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaFreeHost(void *ptr) {
    auto ipc_cli = IPCClient::get_client();
    CudaFreeHostMsg msg{MSG_CUDA_FREE_HOST, uint(pid << 16) + 0xffff, ptr};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaMemset(void *devPtr, int value, size_t count) {
    auto ipc_cli = IPCClient::get_client();
    CudaMemsetMsg msg{MSG_CUDA_MEMSET, uint(pid << 16) + 0xffff, devPtr, value, count};
    return ipc_cli->send(&msg);
}

bool mgpu::cudaMemcpy(void *dst, const void * src, size_t count, ::cudaMemcpyKind kind) {
    auto ipc_cli = IPCClient::get_client();
    CudaMemcpyMsg msg{MSG_CUDA_MEMCPY, uint(pid << 16) + 0xffff, dst, src, count, kind};
    return ipc_cli->send(&msg);
}
