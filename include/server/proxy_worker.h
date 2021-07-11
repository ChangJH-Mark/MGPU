//
// Created by root on 2021/7/3.
//
#pragma once
#include <thread>
#include <pthread.h>
#include <string>
#include <unistd.h>
namespace mgpu {
    class ProxyWorker : public std::thread {
    public:
        ProxyWorker() = delete;
        ProxyWorker(const ProxyWorker&) = delete;
        ProxyWorker(ProxyWorker &&) = delete;
        explicit ProxyWorker(uint conn, pid_t pid) : m_conn(conn), m_p(pid), std::thread(&ProxyWorker::work, this) {
            pthread_setname_np(native_handle(), "proxy_worker");
            pipe(pipefd);
        }
    public:
        void stop(){
            write(pipefd[1], "stop", 4);
            close(pipefd[1]);
            if(joinable())
                this->join();
        };

    private:
        uint m_conn; // connection
        pid_t m_p;     // with pid
        int pipefd[2];   // stop channel
        void work();
    };
}