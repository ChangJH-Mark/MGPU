//
// Created by root on 2021/3/12.
//

#include <cuda_runtime.h>
#include "server/server.h"
#include "server/device.h"
#include "server/kernel.h"
#include "common/helper.h"
#include "common/Log.h"

using namespace mgpu;

void Device::init() {
    cudaCheck(cudaGetDeviceCount(&num));
    if (num <= 0) {
        std::cerr << "no GPU availiable, please check again! " << std::endl;
        exit(EXIT_FAILURE);
    }
    gpu_list.resize(num);
    gpu_load.resize(num);
    // loop for every gpu
    for (int dev = 0; dev < num; dev++) {
        cudaSetDevice(dev);
        cudaSetDeviceFlags(cudaDeviceBlockingSync | cudaDeviceMapHost);
        auto gpu = new GPU;
        init_gpu(gpu, dev);
        gpu_list[dev] = gpu;

        auto load = new Load;
        gpu_load[dev] = load;
    }
}

const Device::GPU *Device::getDev(int gpuid) {
    if (gpuid >= gpu_list.size())
        return nullptr;
    return gpu_list[gpuid];
}

void Device::destroy() {
    dout(LOG) << "start destroy Device Module" << dendl;
    for (auto item : gpu_list) {
        delete item;
    }

    for (auto load : gpu_load) {
        delete load;
    }
}

int Device::GetBestDev(const mgpu::Task &t) {
    if (num <= 0) {
        std::cerr << "no GPU availiable, please check again! " << std::endl;
        exit(EXIT_FAILURE);
    }
    if (num == 1)
        return 0;

    auto k = KERNELMGR->GetKernel(t.kernel);
    uint warps = (t.conf.block.x * t.conf.block.y * t.conf.block.z + 31) / 32 * t.conf.grid.x * t.conf.grid.y *
                 t.conf.grid.z;
    uint copy2dev = 0;

    // host to device malloc
    for (int i = 0; i < t.hdn; i++) {
        copy2dev += t.hds[i];
    }

    double shortest = 0;
    int res = 0;
    for (int dev = 0; dev < num; dev++) {
        // get spin lock
        auto gpu = gpu_list[dev];
        while (!gpu_load[dev]->spin.try_lock());

        uint totalCopy2dev = gpu_load[dev]->copy2dev + copy2dev;
        uint totalInsts = gpu_load[dev]->insts + warps * k.insts_per_warp;
        double totalMemCycles = gpu_load[dev]->memCycles +
                                (warps * k.memTrans_per_warp * k.aveBytes_per_trans) * 1.0 / (gpu->gmem_max_tp * 1000.0);

        double clocks = (totalCopy2dev * 1.0 / 1000 * 1000) / (8 * 1000) *
                        gpu->gpu_clock; // kGPU cycles costed at memcpy
        clocks += (totalInsts) / (gpu->max_warps * gpu->sms * 4 * 1.0) / 1000; // kGPU cycles costed at insts
        clocks += totalMemCycles; // kGPU cycles costed at memTrans

        if (shortest == 0 || clocks < shortest) {
            shortest = clocks;
            res = dev;
        }
        gpu_load[dev]->spin.unlock();
    }
    gpu_load[res]->copy2dev += copy2dev;
    gpu_load[res]->insts += warps * k.insts_per_warp;
    gpu_load[res]->memCycles +=
            (warps * k.memTrans_per_warp * k.aveBytes_per_trans * 1.0) / (gpu_list[res]->gmem_max_tp * 1000.0);
    return res;
}

void Device::ReleaseDev(int dev, const mgpu::Task &t) {
    while (!gpu_load[dev]->spin.try_lock());
    auto load = gpu_load[dev];
    auto gpu = gpu_list[dev];

    auto k = KERNELMGR->GetKernel(t.kernel);
    uint warps = (t.conf.block.x * t.conf.block.y * t.conf.block.z + 31) / 32 * t.conf.grid.x * t.conf.grid.y *
                 t.conf.grid.z;

    uint copy2dev = 0;
    for (int i = 0; i < t.hdn; i++)
        copy2dev += t.hds[i];

    load->copy2dev -= copy2dev;
    load->insts -= warps * k.insts_per_warp;
    load->memCycles -= (warps * k.memTrans_per_warp * k.aveBytes_per_trans) * 1.0 / (gpu->gmem_max_tp * 1000.0);
    load->spin.unlock();
}

void Device::init_gpu(GPU *gpu, uint id) {
    cudaDeviceProp dev_prop{};
    cudaCheck(cudaGetDeviceProperties(&dev_prop, id));
    gpu->ID = id;
    gpu->sms = dev_prop.multiProcessorCount;
    gpu->regs = dev_prop.regsPerMultiprocessor;
    gpu->share_mem = dev_prop.sharedMemPerMultiprocessor;
    gpu->global_mem = dev_prop.totalGlobalMem;
    gpu->const_mem = dev_prop.totalConstMem;
    gpu->max_blocks = dev_prop.maxBlocksPerMultiProcessor;
    gpu->warp_size = dev_prop.warpSize;
    gpu->max_warps = dev_prop.maxThreadsPerMultiProcessor / gpu->warp_size;
    gpu->gpu_clock = dev_prop.clockRate;
    gpu->gmem_max_tp = 1.0 * (dev_prop.memoryClockRate * dev_prop.memoryBusWidth) / (8 * dev_prop.clockRate);
}

void Device::run() {
    observe();
}

void Device::observe() {
    using std::cout;
    cout << "total GPU number: " << num << endl;
    cout << "--------------------------------" << endl;
    for (auto &iter : gpu_list) {
        cout << "device NO." << iter->ID << endl;
        cout << "max blocks per sm: " << iter->max_blocks << endl;
        cout << "max warps per sm: " << iter->max_warps << endl;
        cout << "warp p_size : " << iter->warp_size << " threads" << endl;
        cout << "stream multiprocessor number : " << iter->sms << endl;
        cout << "registers per sm: " << iter->regs << endl;
        cout << "share memory per sm(bytes) : " << iter->share_mem << endl;
        cout << "global_memory(MiB) : " << (iter->global_mem >> 10) << endl;
        cout << "const memory(MiB) : " << (iter->const_mem >> 10) << endl;
        cout << "gpu clock is(kHz) " << iter->gpu_clock << endl;
        cout << "global mem max throughput : " << (iter->gmem_max_tp) << endl;
        cout << "--------------------------------" << endl;
    }
}