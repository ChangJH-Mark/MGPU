//
// Created by root on 2021/3/16.
//
#include <iostream>
#include <unistd.h>
#include <chrono>
#include "client/api.h"

#define N (1 << 24)

using namespace std;

int main() {
    void * dev_ptr1 = mgpu::cudaMalloc(N);
    void * dev_ptr2 = mgpu::cudaMalloc(N);
    mgpu::cudaMemset(dev_ptr1, 0x1, N);
    mgpu::cudaMemset(dev_ptr2, 0x0, N);
    void * host_ptr = mgpu::cudaMallocHost(N);
    printf("%x", host_ptr);
    mgpu::cudaLaunchKernel({{1024},{1024}, 0, 0}, "vecAdd", dev_ptr1, dev_ptr2, N);
    sleep(5);
    mgpu::cudaMemcpy(host_ptr, dev_ptr2, 10, cudaMemcpyDeviceToHost);
    mgpu::cudaFree(dev_ptr1);
    mgpu::cudaFree(dev_ptr2);
    mgpu::cudaFreeHost(host_ptr);
    return 0;
}
