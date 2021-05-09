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
#ifndef MAX_STREAMS
#define MAX_STREAMS 32
#endif

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
        }GPU;

        int num = 0;
        std::list<GPU*> gpu_list;
        std::map<uint, std::array<cudaStream_t, MAX_STREAMS>> gpu_streams;

    public:
        std::array<cudaStream_t, MAX_STREAMS> getStream(uint device)
        {
            return gpu_streams[device];
        }
        int counts() const {return num;}
    public:
        Device() =default;
        void observe();
        void run() override{};
        void init() override;
        void join() override{};
        void destroy() override;

    private:
        void init_gpu(GPU*, uint id);
    };
}
#endif //FASTGPU_GPU_PROPERTY_H
