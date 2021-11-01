//
// Created by root on 2021/3/11.
//

#ifndef FASTGPU_GPU_PROPERTY_H
#define FASTGPU_GPU_PROPERTY_H

#include <thread>
#include <vector>
#include <map>
#include <unistd.h>
#include "mod.h"

namespace mgpu {
    class Device : public Module {
    public:
        typedef struct GPU {
            uint ID;
            uint sms;
            uint regs;
            uint share_mem;
            uint global_mem;
            uint const_mem;
            uint max_blocks;
            uint max_warps;
            uint warp_size;
            double gmem_max_tp; /* max bytes read / write per gpu clock */
        } GPU;

    public:
        int counts() const { return num; }

    public:
        Device() = default;

        Device(const Device &) = delete;

        Device(const Device &&) = delete;

        const GPU* getDev(int gpuid);

        void observe();

        void run() override {};

        void init() override;

        void join() override {};

        void destroy() override;

        ~Device() override {}

    private:
        int num = 0;
        std::vector<GPU *> gpu_list;

        void init_gpu(GPU *, uint id);
    };
}
#endif //FASTGPU_GPU_PROPERTY_H
