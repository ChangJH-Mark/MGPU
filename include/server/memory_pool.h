//
// Created by root on 2021/8/14.
//

#ifndef FASTGPU_MEMORY_POOL_H
#define FASTGPU_MEMORY_POOL_H

#include "/home/mark/Codes/final_graduation/src/cump/include/interface/gpu_memory_pool.hpp"
#include "/home/mark/Codes/final_graduation/include/cump/cump.hpp"
#include "mod.h"
#include <cuda_runtime.h>
#include <map>

namespace mgpu {

    class MemPool : public Module {
    public:
        virtual void init() {
            gpuPool = new cump::BinningDmAlloc(new cump::CudaMallocWrapper, sizeof(double));
            cpuPool = new cump::BinningHmAlloc(new cump::ShareMemWrapper, sizeof(double));
        };

        virtual void run() {}

        virtual void join() {}

        virtual void destroy() {
            dout(LOG) << "start destroy MemPool Module" << dendl;
            std::lock_guard<std::mutex> lock(cpu_lock);
            for (auto pair: shm_ids) {
                shmdt(pair.first);
                shmctl(pair.second, IPC_RMID, nullptr);
            }
        }

        ~MemPool() override {}

    public:
        void gpuMemoryAlloc(int device, void **ptr, size_t size, cudaStream_t stream) {
            lock_guard<std::mutex> lock(gpu_lock);
            cudaSetDevice(device);
            *ptr = gpuPool->allocate(size, stream);
            mem_region[*ptr] = size;
        };

        void gpuMemoryDeAlloc(int device, void *dev_ptr, cudaStream_t stream) {
            lock_guard<std::mutex> lock(gpu_lock);
            cudaSetDevice(device);
            gpuPool->deallocate(dev_ptr, mem_region[dev_ptr], stream);
            mem_region.erase(dev_ptr);
        }

        void *cpuMemoryAlloc(size_t size) {
            lock_guard<std::mutex> lock(cpu_lock);
            void *res = cpuPool->allocator_->allocate(size);
            if (res != nullptr) {
                shm_ids[res] = *(int *) res;
            }
            return res;
        }

        void cpuMemoryDeAlloc(void *ptr) {
            lock_guard<std::mutex> lock(cpu_lock);
            if (shm_ids.count(ptr) == 0) {
                perror("dealloc share memory not exist");
                exit(EXIT_FAILURE);
            }
            *(int*)ptr = shm_ids[ptr];
            cpuPool->allocator_->deallocate(ptr);
            shm_ids.erase(ptr);
        }

    private:
        // GPU memory region
        std::map<void *, size_t> mem_region; // mem pool needs record address and size, I can't understand through
        // CPU share memory map
        std::map<void *, int> shm_ids;
        std::mutex gpu_lock, cpu_lock;
        cump::BinningDmAlloc *gpuPool;
        cump::BinningHmAlloc *cpuPool;
    };
}

#endif //FASTGPU_MEMORY_POOL_H
