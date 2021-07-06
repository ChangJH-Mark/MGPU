//
// Created by root on 2021/7/3.
//

#include "server/proxy_worker.h"
#include "server/conductor.h"
#include "server/server.h"
#include "server/commands.h"
#include <sys/epoll.h>
#include <memory>

#ifndef MAX_MSG_SIZE
#define MAX_MSG_SIZE (1 << 12) // 4KB
#endif

using namespace mgpu;

void ProxyWorker::work() {
    auto epfd = epoll_create(3);
    struct epoll_event e_tmp{};
    e_tmp.data.fd = pipefd[0];
    e_tmp.events = EPOLLIN;
    epoll_ctl(epfd, EPOLL_CTL_ADD, pipefd[0], &e_tmp);

    e_tmp.data.fd = m_conn;
    e_tmp.events = EPOLLIN | EPOLLRDHUP | EPOLLET;
    epoll_ctl(epfd, EPOLL_CTL_ADD, m_conn, &e_tmp);

    bool stop = false;

    while (!stop) {
        struct epoll_event es[3];
        auto len = epoll_wait(epfd, es, 3, -1);
        for (int i = 0; i < len; i++) {
            if(es[i].data.fd == pipefd[0] /* stop signal */) {
                stop = true;
                break;
            }
            else if(es[i].data.fd == m_conn) {
                if(es[i].events & EPOLLRDHUP /* closed */) {
                    close(pipefd[1]);
                    stop = true;
                    break;
                }
                else if(es[i].events & EPOLLIN /* request */)
                {
                    char * msg = new char[MAX_MSG_SIZE];
                    size_t size = recv(m_conn, msg, MAX_MSG_SIZE, 0);
                    msg[size] = 0;
                    auto cmd = std::make_shared<Command>((AbMsg*)msg, m_conn);
                    CONDUCTOR->conduct(cmd);
                }
            }
        } // for
    } // while
    close(pipefd[0]);
    close(epfd);
    close(m_conn);
}