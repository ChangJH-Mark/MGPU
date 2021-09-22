//
// Created by root on 2021/3/16.
//
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include <future>
#include <sys/ipc.h>
#include <sys/types.h>
#include <sys/shm.h>
#include "common/IPC.h"
#include "common/message.h"

using namespace mgpu;

pid_t mgpu::pid = getpid();

std::shared_ptr<IPCClient> IPCClient::get_client() {
    if (!single_instance) {
        single_instance = std::make_shared<IPCClient>();
    }
    return single_instance;
}

int IPCClient::connect() {
    if (conn != -1)
        return conn;
    conn = ::socket(PF_LOCAL, SOCK_STREAM, 0);
    struct sockaddr_un server_addr{PF_LOCAL};
    strcpy(server_addr.sun_path, server_path);
    if (0 > ::connect(conn, (struct sockaddr *) (&server_addr), SUN_LEN(&server_addr))) {
        ::perror("fail to connect to server:");
        ::exit(1);
    }
    return conn;
}

void IPCClient::socket_send(uint cli, void *msg, size_t size, uint flag, const char *err_msg) {
    if (size != ::send(cli, msg, size, flag)) {
        perror(err_msg);
        exit(1);
    }
}

void IPCClient::socket_recv(uint cli, void *dst, size_t size, uint flag, const char *err_msg) {
    uint recved;
    if (size != (recved = ::recv(cli, dst, size, flag))) {
        printf("want %d bytes, got %d bytes\n", size, recved);
        perror(err_msg);
        exit(1);
    }
}

int IPCClient::send(CudaGetDeviceCountMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaGetDeviceCountMsg), 0, "fail to send cudaGetDeviceCount message");
    int ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaGetDeviceCount return");
    return ret;
}

void *IPCClient::send(CudaMallocMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaMallocMsg), 0, "fail to send cudaMalloc message");
    void *ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaMalloc return");
    return ret;
}

void *IPCClient::send(CudaMallocHostMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaMallocHostMsg), 0, "fail to send cudaMallocHost message");
    mgpu::CudaMallocHostRet ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaMallocHost return");
    auto addr = shmat(ret.shmid, ret.ptr, 0);
    if (ret.ptr != addr) {
        printf("err %s, return addr %lx, attached %lx, shm_id %d\n", strerror(errno), ret.ptr, addr, ret.shmid);
        perror("share memory with different address");
        exit(1);
    }
    return ret.ptr;
}

bool IPCClient::send(CudaFreeMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaFreeMsg), 0, "fail to send cudaFree message");
    bool ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaFree return");
    return ret;
}

bool IPCClient::send(CudaFreeHostMsg *msg) {
    if (0 > shmdt(msg->ptr)) {
        perror("fail to release share memory");
        exit(1);
    }
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaFreeHostMsg), 0, "fail to send cudaFreeHost message");
    bool ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaFreeHost return");
    return ret;
}

bool IPCClient::send(CudaMemsetMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaMemsetMsg), 0, "fail to send cudaMemset message");
    bool ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaMemset return");
    return ret;
}

bool IPCClient::send(CudaMemcpyMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaMemcpyMsg), 0, "fail to send cudaMemcpy message");
    bool ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaMemcpy return");
    return ret;
}

bool IPCClient::send(CudaLaunchKernelMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaLaunchKernelMsg), 0, "fail to send cudaLaunchKernel message");
    bool ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaLaunchKernel return");
    return ret;
}

bool IPCClient::send(CudaStreamCreateMsg *msg, stream_t *stream) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaStreamCreateMsg), 0, "fail to send cudaStreamCreate message");
    socket_recv(cli, stream, sizeof(stream_t), 0, "error to receive cudaStreamCreate return");
    return true;
}

bool IPCClient::send(CudaStreamSyncMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaStreamSyncMsg), 0, "fail to send cudaStreamSynchronize message");
    bool ret;
    socket_recv(cli, &ret, sizeof(bool), 0, "error to receive cudaStreamSynchronize return");
    return ret;
}

bool IPCClient::send(CudaEventCreateMsg *msg, event_t *event) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaEventSyncMsg), 0, "fail to send cudaEventCreate message");
    socket_recv(cli, event, sizeof(event_t), 0, "error to receive cudaEventCreate return");
    return true;
}

bool IPCClient::send(CudaEventDestroyMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaEventDestroyMsg), 0, "fail to send cudaEventDestroy message");
    bool ret;
    socket_recv(cli, &ret, sizeof(bool), 0, "error to receive cudaEventDestroy return");
    return ret;
}

bool IPCClient::send(CudaEventRecordMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaEventRecordMsg), 0, "fail to send cudaEventRecord message");
    bool ret;
    socket_recv(cli, &ret, sizeof(bool), 0, "error to receive cudaEventRecord return");
    return ret;
}

bool IPCClient::send(CudaEventSyncMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaEventSyncMsg), 0, "fail to send cudaEventSync message");
    bool ret;
    socket_recv(cli, &ret, sizeof(bool), 0, "error to receive cudaEventSync return");
    return ret;
}

bool IPCClient::send(CudaEventElapsedTimeMsg *msg, float *ms) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaEventElapsedTimeMsg), 0, "fail to send cudaEventElapsedTime message");
    socket_recv(cli, ms, sizeof(float), 0, "error to receive cudaEventElapsedTime return");
    return true;
}

std::future<void *> IPCClient::send(MatrixMulMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(MatrixMulMsg), 0, "fail to send MatrixMulGPU message");
    auto func = [cli, ipc = single_instance]() -> void * {
        CudaMallocHostRet ret;
        ipc->socket_recv(cli, &ret, sizeof(ret), 0, "error to receive MatrixMulGPU return");
        if (ret.ptr != shmat(ret.shmid, ret.ptr, 0)) {
            perror("share memory with different address");
            return nullptr;
        }
        return ret.ptr;
    };
    return std::async(func);
}

std::future<MulTaskRet> IPCClient::send(MulTaskMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(MulTaskMsg), 0, "fail to send MulTaskMulGPU message");
    auto func = [cli, ipc = single_instance]() -> MulTaskRet {
        MulTaskRet ret;
        ipc->socket_recv(cli, &ret, sizeof(ret), 0, "error to receive MulTaskMulGPU return");
        return ret;
    };
    return std::async(func);
}