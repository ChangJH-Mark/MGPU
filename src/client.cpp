//
// Created by root on 2021/3/16.
//

#include "client/api.h"
#include "common/IPC.h"
using namespace mgpu;
void* mgpu::cudaMalloc(size_t size) {
    auto ipc_cli = IPCClient::get_client();
    ipc_cli->connect();
    cudaMallocMSG msg{MSG_CUDA_MALLOC,uint(pid << 16) + 0xffff, size};
    return ipc_cli->send(&msg);
}