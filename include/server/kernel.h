//
// Created by root on 2021/10/31.
//

#ifndef FASTGPU_KERNELS_H
#define FASTGPU_KERNELS_H

#include "server/mod.h"
#include "server/device.h"
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

    class KernelInstance {
    public:
        explicit KernelInstance(CudaLaunchKernelMsg *msg, int gpuid);

        KernelInstance() = delete;

        KernelInstance(const KernelInstance &) = delete;

        KernelInstance(const KernelInstance &&) = delete;

        void operator=(KernelInstance) = delete;

    public:
        void init();            // init run time configs, default occupancy all
        void launch();          // launch this Kernel
        void sync();            // sync this kernel
        void print_runinfo();
        bool is_finished() {return finished;}        // indicate if Kernel Instance stopped

        void occupancy_all(stream_t ctrl);

        void set_config(int sm_low, int sm_high, int wlimit, stream_t ctrl);    // dynamic set resource configs
        int get_config();                                                       // get resource configs
        void get_runinfo(stream_t ctrl);                                        // get kernel run time stage info

    private:
        bool finished;          // signal variable, if kernel finished
        
        // parameter
        Kernel prop;
        std::string name;
        char param_buf[1024];
        size_t p_size;

        // Kernel device, grid, block, stream
        const Device::GPU *gpu;
        dim3 block, grid;
        dim3 grid_v1;
        cudaStream_t stream;

        // cuda mod & func
        CUmodule mod;
        CUfunction func;
        CUfunction func_v1;
        CUdeviceptr devConf; // runtime config

        // cpu conf
        int max_block_per_sm;
        int *cpuConf;
        int cbytes; /* cpu conf bytes */
    public:
        ~KernelInstance();

        friend class Scheduler;
    };
}

#endif //FASTGPU_KERNELS_H
