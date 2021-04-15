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

const char * mgpu::server_path = "/tmp/mgpu/server.sock";

pid_t mgpu::pid = getpid();

IPCClient* IPCClient::single_instance = nullptr;

IPCClient* IPCClient::get_client() {
    if(single_instance == nullptr){
        single_instance = new IPCClient();
        atexit(destroy_client);
    }
    return single_instance;
}

IPCClient::IPCClient() {
}

uint IPCClient::connect() {
    // local socket
    auto socket = ::socket(PF_LOCAL, SOCK_STREAM, 0);
    // remote socket
    struct sockaddr_un server_addr {PF_LOCAL};
    strcpy(server_addr.sun_path, server_path);
    if(0 > ::connect(socket, (struct sockaddr*)(&server_addr), SUN_LEN(&server_addr)))
    {
        ::perror("fail to connect to server:");
        ::exit(1);
    }
    return socket;
}

void IPCClient::socket_clear(uint socket) {
    struct sockaddr_un addr;
    socklen_t len;
    getsockname(socket, (struct sockaddr*) &addr, &len);
    ::close(socket);
    ::unlink(addr.sun_path);
}

void IPCClient::socket_send(uint cli, void *msg, size_t size, uint flag, const char* err_msg) {
    if(size != ::send(cli, msg, size, flag)) {
        perror(err_msg);
        exit(1);
    }
}

void IPCClient::socket_recv(uint cli, void *dst, size_t size, uint flag, const char *err_msg) {
    uint recved;
    if(size != (recved = ::recv(cli, dst, size, flag))) {
        printf("want %d bytes, got %d bytes\n", size, recved);
        perror(err_msg);
        exit(1);
    }
}

void* IPCClient::send(CudaMallocMsg* msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaMallocMsg), 0, "fail to send cudaMalloc message");
    void * ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaMalloc return");
    std::async(&IPCClient::socket_clear, this, cli);
    return ret;
}

void* IPCClient::send(CudaMallocHostMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaMallocHostMsg), 0, "fail to send cudaMallocHost message");
    mgpu::CudaMallocHostRet ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaMallocHost return");
    std::async(&IPCClient::socket_clear, this, cli);
    if(ret.ptr != shmat(ret.shmid,ret.ptr, 0)) {
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
    std::async(&IPCClient::socket_clear, this, cli);
    return ret;
}

bool IPCClient::send(CudaFreeHostMsg *msg) {
    if(0 > shmdt(msg->ptr)) {
        perror("fail to release share memory");
        exit(1);
    }
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaFreeHostMsg), 0, "fail to send cudaFreeHost message");
    bool ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaFreeHost return");
    std::async(&IPCClient::socket_clear, this, cli);
    return ret;
}

bool IPCClient::send(CudaMemsetMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaMemsetMsg), 0, "fail to send cudaMemset message");
    bool ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaMemset return");
    std::async(&IPCClient::socket_clear, this, cli);
    return ret;
}

bool IPCClient::send(CudaMemcpyMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaMemcpyMsg), 0, "fail to send cudaMemcpy message");
    bool ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaMemcpy return");
    std::async(&IPCClient::socket_clear, this, cli);
    return ret;
}

bool IPCClient::send(CudaLaunchKernelMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaLaunchKernelMsg), 0, "fail to send cudaLaunchKernel message");
    bool ret;
    socket_recv(cli, &ret, sizeof(ret), 0, "error to receive cudaLaunchKernel return");
    std::async(&IPCClient::socket_clear, this, cli);
    return ret;
}

bool IPCClient::send(CudaStreamCreateMsg *msg, int * streams) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaStreamCreateMsg), 0, "fail to send cudaStreamCreate message");
    socket_recv(cli, streams, msg->num * sizeof(int), 0, "error to receive cudaStreamCreate return");
    std::async(&IPCClient::socket_clear, this, cli);
    return true;
}

bool IPCClient::send(CudaStreamSyncMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(CudaStreamSyncMsg), 0, "fail to send cudaStreamSynchronize message");
    bool ret;
    socket_recv(cli, &ret, sizeof(bool), 0, "error to receive cudaStreamSynchronize return");
    std::async(&IPCClient::socket_clear, this, cli);
    return ret;
}

std::future<void*> IPCClient::send(MatrixMulMsg *msg) {
    auto cli = connect();
    socket_send(cli, msg, sizeof(MatrixMulMsg), 0, "fail to send MatrixMulGPU message");
    auto func = [cli, ipc = IPCClient::single_instance]() -> void * {
        CudaMallocHostRet ret;
        ipc->socket_recv(cli, &ret, sizeof(ret), 0, "error to receive MatrixMulGPU return");
        ipc->socket_clear(cli);
        if(ret.ptr != shmat(ret.shmid,ret.ptr, 0)) {
            perror("share memory with different address");
            return nullptr;
        }
        return ret.ptr;
    };
    return std::move(std::async(func));
}

IPCClient::~IPCClient() {
}

void mgpu::destroy_client() {
    delete IPCClient::get_client();
}