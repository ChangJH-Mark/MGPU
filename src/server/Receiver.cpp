//
// Created by root on 2021/3/13.
//

#include "server/receiver.h"
#include "server/server.h"
#include "common/IPC.h"
#include <unistd.h>
#include <sys/un.h>
#include <iostream>
#include <cuda_runtime.h>

using namespace mgpu;

void Receiver::init() {
    const char* socket_address = mgpu::server_path;
    max_worker = 100;
    num_worker = 0;
    server_socket = socket(PF_LOCAL, SOCK_STREAM, 0);
    if(server_socket < 0){
        perror("fail to create server sock");
        exit(1);
    }
    struct sockaddr_un server_address {AF_LOCAL};
    strcpy(server_address.sun_path, socket_address);
    if(0 > bind(server_socket, (struct sockaddr*) &server_address, SUN_LEN(&server_address))) {
        perror("fail to bind server socket");
        exit(1);
    }
    if(0 > listen(server_socket, 100)){
        perror("fail to listen server socket");
        exit(1);
    }
}

void Receiver::destroy() {
    close(server_socket);
    unlink(mgpu::server_path);
}

void Receiver::run() {
    this->listener = std::move(std::thread(&Receiver::do_accept, this, server_socket));
}

void Receiver::do_worker(uint socket, struct sockaddr* cli, socklen_t* len) {
    auto msg = (void *)malloc(256);
    auto size = recv(socket, msg, 256, 0);
    *((char*)msg + size) = 0;
    message_t type = *(message_t*)msg;
    switch (type) {
        case MSG_CUDA_MALLOC: {
            push_command(static_cast<cudaMallocMSG*>(msg));
            break;
        }
        default:
            std::cout << "fail to recognize message!" << std::endl;
    }
    num_worker--;
}

// push commands to server task list
// thread-safe
void Receiver::push_command(AbMSG *msg) {
    auto server = get_server();
    server->mtx.lock();
    Command cmd(msg);
    server->cmd_list.push_back(Command(msg));
    std::cout << "now command list size is " << server->cmd_list.size() << std::endl;
    server->mtx.unlock();
}

void Receiver::do_accept(uint socket) {
    while(1){
        if(num_worker >= max_worker){
            continue;
        }
        auto cli = (struct sockaddr_un *)malloc(sizeof(struct sockaddr_un));
        auto len = (socklen_t*)malloc(sizeof(socklen_t));
        auto conn_sock = accept(socket,(struct sockaddr*) cli, len);
        if(conn_sock < 0){
            printf("fail to connect with client");
            free(cli);
            free(len);
        }
        auto worker = std::thread(&Receiver::do_worker, this, conn_sock, (struct sockaddr*) cli, len);
        worker.detach();
        num_worker++;
    }
}

void Receiver::join() {
    listener.join();
}