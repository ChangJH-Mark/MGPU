//
// Created by root on 2021/3/16.
//
#include <string.h>
#include <unistd.h>
#include <stdio.h>
#include "common/IPC.h"
#include "common/message.h"
using namespace mgpu;

const char * mgpu::server_path = "/tmp/mgpu/server.sock";

IPCClient* IPCClient::single_instance = nullptr;

IPCClient* IPCClient::get_client() {
    if(single_instance == nullptr){
        pid_t pid = getpid();
        char exe[256];
        auto size = readlink("/proc/self/exe", exe, 256);
        exe[size] = 0;
        std::string exe_str(exe);
        auto index = exe_str.rfind("/");
        std::string path = exe_str.substr(index, exe_str.length() - index) + "_" + std::to_string(pid);
        single_instance = new IPCClient(path);
    }
    return single_instance;
}

IPCClient::IPCClient(const std::string &name){
    this->socket = ::socket(PF_LOCAL, SOCK_STREAM, 0);
    this->address = "/tmp/mgpu/" + name + ".sock";
    struct sockaddr_un sock_addr{PF_LOCAL};
    strcpy(sock_addr.sun_path,this->address.c_str());
    if(0 > ::bind(this->socket, (struct sockaddr *)&sock_addr, SUN_LEN(&sock_addr))) {
        ::perror("fail to initialize ipc client");
        ::exit(1);
    }
}

void IPCClient::connect() {
    struct sockaddr_un server_addr {PF_LOCAL};
    strcpy(server_addr.sun_path, server_path);
    if(0 > ::connect(this->socket, (struct sockaddr*)(&server_addr), SUN_LEN(&server_addr)))
    {
        ::perror("fail to connect to server:");
        exit(1);
    }
}

void* IPCClient::send(cudaMallocMSG* msg) {
    auto size = sizeof(cudaMallocMSG);
    if(size !=::send(this->socket, msg, size, 0)){
        perror("fail to send cudaMalloc message");
        exit(1);
    }
    void * ret;
    auto ret_size = sizeof(void *);
    if(ret_size !=::recv(this->socket, (void *)&ret, ret_size, 0)) {
        perror("error to receive cudaMalloc return");
        exit(1);
    }
    return ret;
}

IPCClient::~IPCClient() {
    close(socket);
    unlink(this->address.c_str());
}