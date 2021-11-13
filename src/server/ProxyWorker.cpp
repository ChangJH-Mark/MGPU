//
// Created by root on 2021/7/3.
//

#include "server/proxy_worker.h"
#include "server/conductor.h"
#include "server/server.h"
#include "server/commands.h"
#include "common/helper.h"
#include <string.h>
#include <memory>
#include <linux/futex.h>

#ifndef MAX_MSG_SIZE
#define MAX_MSG_SIZE (1 << 12) // 4KB
#endif

using namespace mgpu;

void ProxyWorker::work() {
    pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, nullptr);

    // make sure futex is ok
    while(c_fut.shm_ptr == BAD_UADDR);
    while(s_fut.shm_ptr == BAD_UADDR);

    // start listen to client
    while (!m_stop) {
        if(!c_fut.ready()) {
            int res = syscall(__NR_futex, c_fut.shm_ptr, FUTEX_WAIT, NOT_READY, NULL);
            if(res == -1) {
                if(errno != EAGAIN) {
                    printf("error to recv from client, syscall return %d, errno is %d before read data\n", res, errno);
                    exit(1);
                }
            }
        }
        c_fut.copyTo(buf, c_fut.size());
        buf[c_fut.size()] = '\0';
        c_fut.setState(NOT_READY);

        auto cmd = std::make_shared<Command>((AbMsg *) buf, s_fut.shm_ptr, this);
        CONDUCTOR->conduct(cmd);
    }
}

void ProxyWorker::get_funcs(const std::string& ptx_name, const std::string &kname, CUfunction& func, CUdeviceptr& devPtr, int * & cpuConf, int cbytes) {
    if(mods.find(ptx_name) == mods.end()) {
        KernelMod *mod = new KernelMod;
        mod->cpuConf = new int[cbytes];
        cudaCheck(cuModuleLoad(&mod->mod, ptx_name.c_str()));
        cudaCheck(cuModuleGetGlobal(&mod->devPtr, nullptr, mod->mod, "configs"));
        mod->cpuConf = new int[cbytes];
        cudaCheck(cuModuleGetFunction(&mod->funcs[kname], mod->mod, kname.c_str()));
        mods[ptx_name] = mod;
    }
    auto module = mods[ptx_name];
    cpuConf = module->cpuConf;
    devPtr = module->devPtr;
    if(module->funcs.find(kname) == module->funcs.end()) {
        cudaCheck(cuModuleGetFunction(&module->funcs[kname], module->mod, kname.c_str()));
    }
    func = module->funcs[kname];
}