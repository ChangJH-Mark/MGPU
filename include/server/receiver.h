//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_RECEIVER_H
#define FASTGPU_RECEIVER_H

#include <thread>
#include <map>
#include <sys/socket.h>
#include <sys/epoll.h>
#include "mod.h"
#include "common/message.h"
#include "common/ThreadPool.h"
#include "commands.h"
#include "server/server.h"
#include "server/proxy_worker.h"

namespace mgpu {
    class Receiver : public Module {
    public:
        Receiver() {
            joinable = true;
            max_worker = 16;
        };

        Receiver(const Receiver &) = delete;

        Receiver(const Receiver &&) = delete;

        void init() override;

        void run() override;

        void join() override;

        void destroy() override;

    private:
        void do_accept();

        void do_newconn(uint conn);

    private:
        uint max_worker;
        uint server_socket;
        std::thread listener;
        int epfd;
        int stopfd[2];
        map<uint, shared_ptr<ProxyWorker>> workers;
    };
}
#endif //FASTGPU_RECEIVER_H
