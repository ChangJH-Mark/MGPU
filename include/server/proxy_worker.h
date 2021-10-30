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

        ProxyWorker(const ProxyWorker &) = delete;

        ProxyWorker(ProxyWorker &&) = delete;

        explicit ProxyWorker(void *c, void *s) : c_fut(c), s_fut(s), m_stop(false), buf(new char[PAGE_SIZE]),
                                                 std::thread(&ProxyWorker::work, this) {
        }

    public:
        void stop() {
            m_stop = true;
            pthread_cancel(this->native_handle());
            if (joinable())
                this->join();
            delete[] buf;
        };

    private:
        Futex c_fut, s_fut;
        bool m_stop;
        char *buf;

        void work();

    public:
        std::unordered_map<void *, int> shms_id; // shm id allocated by this proxy
    };
}