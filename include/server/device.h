//
// Created by root on 2021/3/11.
//

#ifndef FASTGPU_GPU_PROPERTY_H
#define FASTGPU_GPU_PROPERTY_H

#include <thread>
#include <vector>
#include <mutex>
#include <map>
#include <unistd.h>
#include "mod.h"
#include "common/message.h"

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
            uint gpu_clock;
            double gmem_max_tp; /* max bytes read / write per gpu clock */
        } GPU;

    public:
        int counts() const { return num; }

    public:
        Device() = default;

        Device(const Device &) = delete;

        Device(const Device &&) = delete;

        const GPU *getDev(int gpuid);

        void observe();

        void run() override;

        void init() override;

        void join() override {};

        void destroy() override;

        ~Device() override {}

    public:
        int GetBestDev(const mgpu::Task &t);
        void ReleaseDev(int dev, const mgpu::Task &t);

    private:
        typedef struct {
            uint copy2dev;      // pending copy to dev bytes
            uint insts;         // pending instructions to execute
            uint memCycles;      // pending memory cycle costs
            std::mutex spin;
        } Load;

        int num = 0;
        std::vector<GPU *> gpu_list;
        std::vector<Load *> gpu_load;  // indicate every gpu load now, including a spin lock

        void init_gpu(GPU *, uint id);
    };
}
#endif //FASTGPU_GPU_PROPERTY_H
