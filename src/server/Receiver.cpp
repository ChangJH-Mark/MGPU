//
// Created by root on 2021/3/13.
//

#include "server/receiver.h"
#include <unistd.h>
#include <sys/un.h>
#include <iostream>
#define SERVER_ADDR "/tmp/mgpu/server.sock"

using namespace mgpu;

void Receiver::init() {
    const char* socket_address = SERVER_ADDR;
    max_worker = 4;
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
    if(0 > listen(server_socket, 10)){
        perror("fail to listen server socket");
        exit(1);
    }
}

void Receiver::destroy() {
    close(server_socket);
    unlink(SERVER_ADDR);
}

void Receiver::run() {
    this->listener = std::move(std::thread(&Receiver::do_listen, this, server_socket));
}

void Receiver::do_worker(uint socket, struct sockaddr* cli, socklen_t* len) {
    char * buffer = new char[256];
    auto ssize = recv(socket, buffer, 256, 0);
    buffer[ssize] = 0;
    std::cout << "thread: " << std::this_thread::get_id() << std::endl;
    std::cout << "message: " << buffer << std::endl;
    delete [] buffer;
    num_worker--;
}

void Receiver::do_listen(uint socket) {
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