//
// Created by root on 2021/7/3.
//
#pragma once

#include <thread>
#include <pthread.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <sys/mman.h>
#include "common/message.h"
#include "common/helper.h"

namespace mgpu {

    typedef struct KernelMod {
        CUmodule mod;
        CUdeviceptr devPtr;
        int *cpuConf;
        std::unordered_map<std::string, CUfunction> funcs;
    }KernelMod;

    class ProxyWorker : public std::thread {
    public:
        ProxyWorker() = delete;

        ProxyWorker(const ProxyWorker &) = delete;

        ProxyWorker(ProxyWorker &&) = delete;

        explicit ProxyWorker(pid_t p, void *c, void *s) : cpid(p), c_fut(c), s_fut(s), m_stop(false),
                                                          buf(new char[PAGE_SIZE]),
                                                          std::thread(&ProxyWorker::work, this) {
        }

    public:
        pid_t get_peer() const { return cpid; }

        void stop() {
            m_stop = true;
            pthread_cancel(this->native_handle());
            if (joinable())
                this->join();
            munmap(c_fut.shm_ptr, PAGE_SIZE);
            munmap(s_fut.shm_ptr, PAGE_SIZE);
            delete[] buf;
            for(auto& mod : mods) {
                cudaCheck(cuModuleUnload(mod.second->mod));
                delete[] mod.second->cpuConf;
                delete mod.second;
            }
        };

    public:
        void get_funcs(const std::string& ptx_name, const std::string &kname, CUfunction& func, CUdeviceptr& devPtr, int * & cpuConf, int cbytes);

    private:
        Futex c_fut, s_fut;
        char *buf;
        pid_t cpid;
        bool m_stop;
        std::unordered_map<std::string, KernelMod*> mods;

        void work();

    public:
        std::unordered_map<void *, int> shms_id; // shm id allocated by this proxy
    };
}