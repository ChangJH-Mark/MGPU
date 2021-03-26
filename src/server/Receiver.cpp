//
// Created by root on 2021/3/13.
//

#include "server/receiver.h"
#include "server/server.h"
#include "common/IPC.h"
#include <unistd.h>
#include <sys/un.h>
#include <iostream>

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
    if(access(mgpu::server_path, F_OK) == 0){
        unlink(mgpu::server_path);
        std::cout << "socket path: " << mgpu::server_path << " already exist, delete" << std::endl;
    }
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
    this->listener = std::move(std::thread(&Receiver::do_accept, this));
    auto handler = this->listener.native_handle();
    pthread_setname_np(handler, "Listener");
}

void Receiver::do_worker(uint socket, struct sockaddr* cli, socklen_t* len) {
    auto msg = (void *)malloc(256);
    auto size = recv(socket, msg, 256, 0);
    *((char*)msg + size) = 0;
    msg_t type = *(msg_t*)msg;
    switch (type) {
        case MSG_CUDA_MALLOC:
        case MSG_CUDA_MALLOC_HOST:
        case MSG_CUDA_FREE:
        case MSG_CUDA_FREE_HOST:
        case MSG_CUDA_MEMSET:
        case MSG_CUDA_MEMCPY:
        case MSG_CUDA_LAUNCH_KERNEL:
            push_command(static_cast<AbMsg *>(msg), socket);
            break;
//        case MSG_CUDA_MALLOC: {
//            push_command(static_cast<CudaMallocMsg*>(msg), socket);
//            break;
//        }
//        case MSG_CUDA_MALLOC_HOST: {
//            push_command(static_cast<CudaMallocHostMsg*>(msg), socket);
//            break;
//        }
        default:
            std::cerr << "fail to recognize message!" << std::endl;
    }
    free(cli);
    free(len);
    num_worker--;
}

// push commands to server task list
// thread-safe
// @msg point to client's request, @cli is the conn socket.
void Receiver::push_command(AbMsg *msg, uint cli) {
    auto server = get_server();
    server->map_mtx.lock();
    if(server->task_map.find(msg->key) == server->task_map.end()) {
        server->task_map[msg->key] = make_pair(make_shared<std::mutex>(), make_shared<Server::List>());
    }
    auto mtx = server->task_map[msg->key].first;
    auto list = server->task_map[msg->key].second;
    mtx->lock();
    list->push_back(make_shared<Command>(msg, cli));
    mtx->unlock();
    std::cout << "now command map p_size is " << server->task_map.size() << std::endl;
    for(auto m : server->task_map){
        std::cout << "pid : " << (m.first >> 16) << std::endl;
        std::cout << "key : " << (m.first) << std::endl;
        std::cout << "list p_size: " << (m.second.second->size()) << std::endl;
    }
    server->map_mtx.unlock();
}

void Receiver::do_accept() {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    while(1){
        if(num_worker >= max_worker){
            continue;
        }
        auto cli = (struct sockaddr_un *)malloc(sizeof(struct sockaddr_un));
        auto len = (socklen_t*)malloc(sizeof(socklen_t));
        auto conn_sock = accept(server_socket, (struct sockaddr*) cli, len);
        if(conn_sock < 0){
            printf("fail to connect with client");
            free(cli);
            free(len);
        }
        std::thread worker(&Receiver::do_worker, this, conn_sock, (struct sockaddr*) cli, len);
        auto handler = worker.native_handle();
        pthread_setname_np(handler, "receiver");
        worker.detach();
        num_worker++;
    }
}

void Receiver::join() {
    listener.detach();
}