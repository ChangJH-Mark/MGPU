//
// Created by root on 2021/3/12.
//

#include <cuda_runtime.h>
#include "server/device.h"
#include "common/helper.h"

using namespace mgpu;

void mgpu::Device::init() {
    cudaCheck(cudaGetDeviceCount(&num));
    if(num <= 0){
        std::cerr << "no GPU availiable, please check again! " << std::endl;
        exit(EXIT_FAILURE);
    }
    // loop for every gpu
    for(int dev = 0; dev < num; dev++){
        ::cudaSetDevice(dev);
        ::cudaSetDeviceFlags(cudaDeviceBlockingSync | cudaDeviceMapHost);
        auto gpu = new GPU;
        init_gpu(gpu, dev);
        gpu_list.push_back(gpu);
        std::array<cudaStream_t, MAX_STREAMS> streams{};
        for(int j = 0; j < MAX_STREAMS; j++){
            cudaCheck(cudaStreamCreate(&streams[j]));
            std::cout << __FUNCTION__ << " create device: " << dev << " stream: " << j << " ptr: " << streams[j] << std::endl;
        }
        gpu_streams[dev] = streams;
    }
}

void mgpu::Device::destroy() {
    for(auto item : gpu_list){
        delete item;
    }
}

void mgpu::Device::init_gpu(GPU *gpu, uint id) {
    cudaDeviceProp dev_prop{};
    cudaCheck(cudaGetDeviceProperties(&dev_prop, id));
    gpu->ID = id;
    gpu->max_blocks = dev_prop.maxThreadsPerMultiProcessor;
    gpu->warp_size = dev_prop.warpSize;
    gpu->sms = dev_prop.multiProcessorCount;
    gpu->share_mem = dev_prop.sharedMemPerMultiprocessor;
    gpu->global_mem = dev_prop.totalGlobalMem;
}

void mgpu::Device::observe() {
    using std::cout;
    cout << "total GPU number: " << num << endl;
    cout << "--------------------------------" << endl;
    for(auto & iter : gpu_list) {
        cout << "device NO." << iter->ID << endl;
        cout << "max blocks per sm: " << iter->max_blocks << endl;
        cout << "warp p_size : " << iter->warp_size << " threads" << endl;
        cout << "stream multiprocessor number : " << iter->sms << endl;
        cout << "share memory per sm(bytes) : " << iter->share_mem << endl;
        cout << "global_memory(MiB) : " << (iter->global_mem >> 10) << endl;
        cout << "--------------------------------" << endl;
    }
}