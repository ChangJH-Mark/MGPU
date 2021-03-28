//
// Created by root on 2021/3/16.
//
#include <iostream>
#include <unistd.h>
#include <chrono>
#include "client/api.h"

#define N (10000)

using namespace std;

int main() {
    void * dev_ptr1 = mgpu::cudaMalloc(N);
    void * dev_ptr2 = mgpu::cudaMalloc(N);
    mgpu::cudaMemset(dev_ptr1, 0x1, N);
    mgpu::cudaMemset(dev_ptr2, 0x2, N);
    void * host_ptr = mgpu::cudaMallocHost(N);
    printf("dev_ptr1: 0x%lx dev_ptr2: 0x%lx, host_ptr: 0x%lx\n",dev_ptr1, dev_ptr2, host_ptr);
    mgpu::cudaLaunchKernel({{1},{1024}, 0, 0}, "vecAdd", dev_ptr1, dev_ptr2, N);
    sleep(10);
    mgpu::cudaMemcpy(host_ptr, dev_ptr2, N, cudaMemcpyDeviceToHost);
    printf("0x%x\n", *(int*)host_ptr);
    mgpu::cudaFree(dev_ptr1);
    mgpu::cudaFree(dev_ptr2);
    mgpu::cudaFreeHost(host_ptr);
    return 0;
}
