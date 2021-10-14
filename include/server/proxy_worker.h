//
// Created by root on 2021/7/3.
//
#pragma once
#include <thread>
#include <pthread.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include "common/message.h"

namespace mgpu {
    class ProxyWorker : public std::thread {
    public:
        ProxyWorker() = delete;
        ProxyWorker(const ProxyWorker&) = delete;
        ProxyWorker(ProxyWorker &&) = delete;
        explicit ProxyWorker(uint conn, pid_t pid) : m_conn(conn), m_p(pid), m_stop(false), std::thread(&ProxyWorker::work, this) {
            buf = new char[PAGE_SIZE];
        }
    public:
        void stop(){
            m_stop = true;
            if(joinable())
                this->join();
            delete[] buf;
        };

    private:
        uint m_conn; // connection
        pid_t m_p;     // with pid
        Futex c_fut, s_fut;
        bool m_stop;
        char *buf;
        void work();
        int init_shm();
        void init_pid();

    public:
        std::unordered_map<void *, int> shms_id; // shm id allocated by this proxy
    };
}