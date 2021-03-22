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
    void * dev_ptr = mgpu::cudaMalloc(N);
    mgpu::cudaMemset(dev_ptr, 0x1, N);
    void * host_ptr = mgpu::cudaMallocHost(N);
    mgpu::cudaMemcpy(host_ptr, dev_ptr, N, cudaMemcpyDeviceToHost);
    sleep(3);
    mgpu::cudaFree(dev_ptr);
    mgpu::cudaFreeHost(host_ptr);
//    auto ret = mgpu::cudaMalloc(1 << 20);
//    cout << "malloc success!" << ret << endl;
//
//    auto h_ret = mgpu::cudaMallocHost(1 << 20);
//    cout << "malloc host success!" << h_ret << endl;
    return 0;
}
