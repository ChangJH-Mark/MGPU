//
// Created by root on 2021/3/11.
//

#ifndef FASTGPU_GPU_PROPERTY_H
#define FASTGPU_GPU_PROPERTY_H

#include <thread>
#include <list>
#include <map>
#include <unistd.h>
#include "mod.h"

namespace mgpu {
    class Device : public Module {
    private:
        typedef struct GPU {
            uint ID;
            uint max_blocks;
            uint warp_size;
            uint sms;
            uint share_mem;
            uint global_mem;
            uint const_mem;
        } GPU;

        int num = 0;
        std::list<GPU *> gpu_list;

    public:
        int counts() const { return num; }

    public:
        Device() = default;

        Device(const Device &) = delete;

        Device(const Device &&) = delete;

        void observe();

        void run() override {};

        void init() override;

        void join() override {};

        void destroy() override;

    private:
        void init_gpu(GPU *, uint id);
    };
}
#endif //FASTGPU_GPU_PROPERTY_H
