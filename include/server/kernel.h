//
// Created by root on 2021/10/31.
//

#ifndef FASTGPU_KERNELS_H
#define FASTGPU_KERNELS_H

#include "server/mod.h"
#include <unordered_map>
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>

namespace mgpu {

    typedef struct {
        float property;
        int regs;
        int shms;
    } Kernel;

    class KernelInstance;
    class KernelMgr : public Module {
    public:
        KernelMgr() = default;

        KernelMgr(const KernelMgr &) = delete;

        KernelMgr(const KernelMgr &&) = delete;

        void init() override;

        void run() override {};

        void join() override {};

        void destroy() override {};

        void obverse();

        ~KernelMgr() override {};
    private:
        std::unordered_map<std::string, Kernel> kns; // kernels
        friend class KernelInstance;
    };

    class KernelInstance{
    public:
        explicit KernelInstance(CudaLaunchKernelMsg* msg, int gpuid);
        KernelInstance() = delete;
        KernelInstance(const KernelInstance &) = delete;
        KernelInstance(const KernelInstance&&) = delete;
        void operator=(KernelInstance) = delete;

    public:
        void init(); // init run time configs
        void run();
        void set_config(); // dynamic set run time configs
        void getInfo(); // get run time info

    private:
        // parameter
        Kernel prop;
        std::string name;
        char param_buf[1024];
        size_t p_size;

        // Kernel device, grid, block, stream
        int dev;
        dim3 block, grid;
        dim3 grid_v1;
        cudaStream_t stream;

        // cuda mod & func
        CUmodule mod;
        CUfunction func;
        CUfunction func_v1;
        CUdeviceptr conf_ptr; // runtime config

        // cpu conf
        int max_block_per_sm;
        int *conf;

        ~KernelInstance();
    };
}

#endif //FASTGPU_KERNELS_H
