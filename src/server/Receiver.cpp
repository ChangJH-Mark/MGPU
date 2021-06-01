//
// Created by root on 2021/3/13.
//

#include "server/receiver.h"
#include "server/server.h"
#include "server/task.h"
#include "common/IPC.h"
#include <unistd.h>
#include <sys/un.h>
#include <iostream>
#define MAX_MSG_SIZE (1 << 12) // 4KB

using namespace mgpu;

void Receiver::init() {
    const char* socket_address = mgpu::server_path;
    server_socket = socket(PF_LOCAL, SOCK_STREAM, 0);
    if(server_socket < 0){
        perror("fail to create server sock");
        exit(1);
    }
    struct sockaddr_un server_address {AF_LOCAL};
    strcpy(server_address.sun_path, socket_address);
    if(access(mgpu::server_path, F_OK) == 0){
        unlink(mgpu::server_path);
        dout(DEBUG) << "socket path: " << mgpu::server_path << " already exist, delete" << dendl;
    }
    if(0 > bind(server_socket, (struct sockaddr*) &server_address, SUN_LEN(&server_address))) {
        perror("fail to bind server socket");
        exit(1);
    }
    if(0 > listen(server_socket, 100)){
        perror("fail to listen server socket");
        exit(1);
    }
    epfd = epoll_create(MAX_CONNECTION);
    struct epoll_event ev{};
    ev.data.fd = server_socket;
    ev.events = EPOLLIN;
    epoll_ctl(epfd, EPOLL_CTL_ADD, server_socket, &ev);
    // stop epoll wait
    pipe(stopfd);
    ev.data.fd = stopfd[0];
    ev.events = EPOLLIN;
    epoll_ctl(epfd, EPOLL_CTL_ADD, stopfd[0], &ev);
}

void Receiver::destroy() {
    close(server_socket);
    unlink(mgpu::server_path);
    write(stopfd[1], "stop", 4);
    close(stopfd[1]);
    stopped = true;
    close(epfd);
    this->listener.join();
}

void Receiver::run() {
    this->listener = std::thread(&Receiver::do_accept, this);
    auto handler = this->listener.native_handle();
    pthread_setname_np(handler, "Listener");
}

void Receiver::push_command(uint conn) {
    char * msg = new char[MAX_MSG_SIZE];
    size_t size = recv(conn, msg, MAX_MSG_SIZE, 0);
    msg[size] = 0;
    auto* api = reinterpret_cast<api_t*>(msg);
    switch (*api) {
        case MSG_CUDA_MALLOC:
        case MSG_CUDA_MALLOC_HOST:
        case MSG_CUDA_FREE:
        case MSG_CUDA_FREE_HOST:
        case MSG_CUDA_MEMSET:
        case MSG_CUDA_MEMCPY:
        case MSG_CUDA_LAUNCH_KERNEL:
        case MSG_CUDA_STREAM_CREATE:
        case MSG_CUDA_STREAM_SYNCHRONIZE:
        case MSG_CUDA_GET_DEVICE_COUNT:
        case MSG_CUDA_EVENT_CREATE:
        case MSG_CUDA_EVENT_DESTROY:
        case MSG_CUDA_EVENT_RECORD:
        case MSG_CUDA_EVENT_SYNCHRONIZE:
        case MSG_CUDA_EVENT_ELAPSED_TIME:
        case MSG_MATRIX_MUL_GPU:
            break;
        default:
            dout(DEBUG) << " fail to recognize message info! " << " api: " <<  *api << " key: " << *(api + 1) << dendl;
            exit(EXIT_FAILURE);
    }
    auto cmd = make_shared<Command>((AbMsg*)msg, conn);
    TASK_HOLDER->insert_cmd(cmd);
}

void Receiver::do_worker(uint conn) {
    pool.commit(&Receiver::push_command, this, conn);
}

void Receiver::do_newconn() {
    uint conn;
    struct ucred cred{};
    struct epoll_event ev{};
    socklen_t len;
    if (0 > (conn = accept(server_socket, nullptr, nullptr))) {
        perror("fail to make connect");
        exit(EXIT_FAILURE);
    }
    getsockopt(conn, SOL_SOCKET, SO_PEERCRED, &cred, &len);
    int base = p_conns.count(cred.pid) ? p_conns[cred.pid] : 0;
    p_conns[cred.pid] = base + 1;
    ev.data.fd = conn, ev.events = EPOLLIN | EPOLLRDHUP | EPOLLET;
    epoll_ctl(epfd, EPOLL_CTL_ADD, conn, &ev);
}

void Receiver::do_close(uint conn) {
    struct ucred cred{};
    socklen_t len;
    getsockopt(conn, SOL_SOCKET, SO_PEERCRED, &cred, &len);
    p_conns[cred.pid]--;
    if(p_conns[cred.pid] == 0){
        p_conns.erase(cred.pid);
        pool.commit(&Task::clear_pid, TASK_HOLDER.get(),cred.pid);
    }
    epoll_ctl(epfd, EPOLL_CTL_DEL, conn, nullptr);
    close(conn);
}

void Receiver::do_accept() {
    std::this_thread::sleep_for(std::chrono::microseconds(100));
    struct epoll_event events[MAX_CONNECTION];
    while (!stopped) {
        int len = epoll_wait(epfd, events, MAX_CONNECTION, -1);
        for (int i = 0; i < len; i++) {
            if (events[i].data.fd == stopfd[0]) {
                dout(DEBUG) << " server close " << stopfd[0] << dendl;
                close(stopfd[0]);
                break;
            } else if (events[i].data.fd == server_socket) {
                dout(DEBUG) << " new connection " << dendl;
                do_newconn();
            } else if (events[i].events & EPOLLRDHUP) {
                dout(DEBUG) << " get EPOLLRDHUP in " << events[i].data.fd << dendl;
                do_close(events[i].data.fd);
            } else if (events[i].events & EPOLLIN) {
                dout(DEBUG) << " get EPOLLIN in " << events[i].data.fd << dendl;
                do_worker(events[i].data.fd);
            } else {
                dout(DEBUG) << " unexpected epoll events happened " << dendl;
            }
        } // while(!stopped)
    }
}

void Receiver::join() {
//    listener.detach();
}