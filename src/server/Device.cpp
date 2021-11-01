//
// Created by root on 2021/3/12.
//

#include <cuda_runtime.h>
#include "server/device.h"
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
    // loop for every gpu
    for (int dev = 0; dev < num; dev++) {
        cudaSetDevice(dev);
        cudaSetDeviceFlags(cudaDeviceBlockingSync | cudaDeviceMapHost);
        auto gpu = new GPU;
        init_gpu(gpu, dev);
        gpu_list[dev] = gpu;
    }
}

const Device::GPU* Device::getDev(int gpuid) {
    if(gpuid >= gpu_list.size())
        return nullptr;
    return gpu_list[gpuid];
}

void Device::destroy() {
    dout(LOG) << "start destroy Device Module" << dendl;
    for (auto item : gpu_list) {
        delete item;
    }
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
    gpu->gmem_max_tp = 1.0 * (dev_prop.memoryClockRate * dev_prop.memoryBusWidth) / (8 * dev_prop.clockRate);
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
        cout << "share memory per sm(bytes) : " << iter->share_mem << endl;
        cout << "global_memory(MiB) : " << (iter->global_mem >> 10) << endl;
        cout << "const memory(MiB) : " << (iter->const_mem >> 10) << endl;
        cout << "global mem max throughput : " << (iter->gmem_max_tp) << endl;
        cout << "--------------------------------" << endl;
    }
}