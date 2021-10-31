//
// Created by root on 2021/10/31.
//

#ifndef FASTGPU_KERNELS_H
#define FASTGPU_KERNELS_H

#include "server/mod.h"
#include <unordered_map>
#include <string>

namespace mgpu {

    typedef struct {
        float property;
    } Kernel;

    class KernelMgr : public Module {
    public:
        KernelMgr() = default;

        KernelMgr(const KernelMgr &) = delete;

        KernelMgr(const KernelMgr &&) = delete;

        void init() override;

        void run() override {};

        void join() override {};

        void destroy() override {};

        ~KernelMgr() override {};
    private:
        std::unordered_map<std::string, Kernel> kns; // kernels
    };

}

#endif //FASTGPU_KERNELS_H
