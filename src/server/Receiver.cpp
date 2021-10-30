//
// Created by root on 2021/3/13.
//

#include "server/receiver.h"
#include "server/server.h"
#include "server/conductor.h"
#include "common/IPC.h"
#include <unistd.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <fcntl.h>

#define MAX_MSG_SIZE (1 << 12) // 4KB

using namespace mgpu;

void Receiver::init() {
    const char *socket_address = mgpu::server_path;
    server_socket = socket(PF_LOCAL, SOCK_STREAM, 0);
    if (server_socket < 0) {
        perror("fail to create server sock");
        exit(1);
    }
    struct sockaddr_un server_address{AF_LOCAL};
    strcpy(server_address.sun_path, socket_address);
    if (access(mgpu::server_path, F_OK) == 0) {
        unlink(mgpu::server_path);
        dout(LOG) << "socket path: " << mgpu::server_path << " already exist, delete" << dendl;
    }
    if (0 > bind(server_socket, (struct sockaddr *) &server_address, SUN_LEN(&server_address))) {
        perror("fail to bind server socket");
        exit(1);
    }
    if (0 > listen(server_socket, 100)) {
        perror("fail to listen server socket");
        exit(1);
    }

    epfd = epoll_create(max_worker + 2); // worker + server_socket + stopfd

    struct epoll_event ev{};
    ev.data.fd = server_socket;
    ev.events = EPOLLIN;
    epoll_ctl(epfd, EPOLL_CTL_ADD, server_socket, &ev);

    // stop epoll wait
    pipe(stopfd);
    ev.data.fd = stopfd[0];
    ev.events = EPOLLIN | EPOLLHUP;
    epoll_ctl(epfd, EPOLL_CTL_ADD, stopfd[0], &ev);
}

void Receiver::destroy() {
    stopped = true;
    // stop listener
    close(stopfd[1]);
    listener.join();

    // stop worker
    for(const auto &w : workers) {
        close(w.first);
        w.second->stop();
    }
    workers.clear();

    // close server socket
    close(server_socket);
    unlink(mgpu::server_path);
}

void Receiver::run() {
    listener = std::thread(&Receiver::do_accept, this);
    auto handler = listener.native_handle();
    pthread_setname_np(handler, "Listener");
}

void init_shm(uint conn, pid_t cpid, void *shms[2]) {
    using std::string;
    string names[2] = {"mgpu.0." + to_string(cpid), "mgpu.1." + to_string(cpid)};
    string root = "/dev/shm/";
    int cnt = 0;
    for (auto &n : names) {
        auto bytes = read(conn, &(shms[cnt]), sizeof(void *));
        assert(bytes == sizeof(void *));
        if (0 != access((root + n).c_str(), F_OK)) {
            printf("error to open shm %s\n", (root + n).c_str());
            exit(EXIT_FAILURE);
        }
        int fd = shm_open(n.c_str(), O_CLOEXEC | O_RDWR, 0644);
        assert(fd > 0);
        if (shms[cnt] != mmap(shms[cnt], PAGE_SIZE, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0)) {
            printf("mgpu mapped at distinct address\n");
            exit(EXIT_FAILURE);
        }
        close(fd);
        cnt++;
    }
}

void Receiver::do_newconn(uint conn) {
    struct ucred ucred{};
    socklen_t len = sizeof(struct ucred);

    if (getsockopt(conn, SOL_SOCKET, SO_PEERCRED, &ucred, &len) < 0) {
        perror("fail to get peer id");
        exit(EXIT_FAILURE);
    }
    void *shms[2] = {nullptr, nullptr};
    init_shm(conn, ucred.pid, shms);

    workers[conn] = make_shared<ProxyWorker>(shms[0], shms[1]);
    auto handler = workers[conn]->native_handle();
    pthread_setname_np(handler, "proxy_worker");
}

void Receiver::do_accept() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    struct epoll_event events[max_worker + 2];
    while (!stopped) {
        int len = epoll_wait(epfd, events, max_worker + 2, -1);
        for (int i = 0; i < len; i++) {
            if (events[i].data.fd == stopfd[0] /* receiver close */) {
                dout(LOG) << " receiver close " << stopfd[0] << dendl;
                close(stopfd[0]);
                stopped = true;
                break;
            } else if (events[i].data.fd == server_socket /* new connection */) {
                dout(LOG) << " new connection " << dendl;

                uint conn = accept(server_socket, nullptr, nullptr);
                if (conn < 0) {
                    perror("fail to make connection");
                    exit(EXIT_FAILURE);
                }

                epoll_event ev{};
                ev.data.fd = conn, ev.events = EPOLLIN | EPOLLRDHUP;
                epoll_ctl(epfd, EPOLL_CTL_ADD, conn, &ev);
                do_newconn(conn);
            } else if (events[i].events & EPOLLRDHUP &&
                       workers.find(events[i].data.fd) != workers.end() /* client close connection */) {
                dout(LOG) << " one client close connection " << dendl;
                auto conn = events[i].data.fd;
                workers[conn]->stop();
                workers.erase(conn);

                epoll_ctl(epfd, EPOLL_CTL_DEL, conn, events + i);
            } else {
                dout(LOG) << " unexpected epoll events happened " << dendl;
                exit(EXIT_FAILURE);
            }
        } // for
    }// while(!stopped)
    close(epfd);
}

void Receiver::join() {
}