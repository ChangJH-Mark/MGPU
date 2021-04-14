//
// Created by root on 2021/3/16.
//
#include <iostream>
#include <unistd.h>
#include <chrono>
#include "client/api.h"

using namespace std;

void test_sm() {
    uint block_num = 10;
    uint size = sizeof(block_num) * block_num;
    void * dev_ptr1 = mgpu::cudaMalloc(size);
    void * host_ptr1 = mgpu::cudaMallocHost(size);
    mgpu::stream_t stream;
    mgpu::cudaStreamCreate(&stream, 1);
    mgpu::cudaLaunchKernel({{block_num},{1}, 0, stream}, "/opt/custom/ptx/specify_sm.ptx", "sm_ids", dev_ptr1);
    mgpu::cudaStreamSynchronize(stream);
    mgpu::cudaMemcpy(host_ptr1, dev_ptr1, size, cudaMemcpyDeviceToHost);
    for(int i=0; i<block_num; i++){
        printf("%d ", *((int*)(host_ptr1) + i));
    }
    printf("\n");
    mgpu::cudaFree(dev_ptr1);
    mgpu::cudaFreeHost(host_ptr1);
}

void test_vecAdd() {
    const int N = 1 << 28;
    void * dev_ptr1 = mgpu::cudaMalloc(N);
    void * dev_ptr2 = mgpu::cudaMalloc(N);
    mgpu::cudaMemset(dev_ptr1, 0x1, N);
    mgpu::cudaMemset(dev_ptr2, 0x2, N);
    void * host_ptr = mgpu::cudaMallocHost(N);
    mgpu::stream_t streams;
    mgpu::cudaStreamCreate(&streams, 1);
    printf("stream: %d\n", streams);
    printf("dev_ptr1: 0x%lx dev_ptr2: 0x%lx, host_ptr: 0x%lx\n",dev_ptr1, dev_ptr2, host_ptr);
    mgpu::cudaLaunchKernel({{200},{10}, 0, streams}, "/opt/custom/ptx/vecAdd.ptx", "vecAdd", dev_ptr1, dev_ptr2, int(N / sizeof(int)));
    mgpu::cudaStreamSynchronize(streams);
    mgpu::cudaMemcpy(host_ptr, dev_ptr2, N, cudaMemcpyDeviceToHost);
    printf("0x%x\n", *((int*)host_ptr + (N-sizeof(int )) / sizeof(int )));
    mgpu::cudaFree(dev_ptr1);
    mgpu::cudaFree(dev_ptr2);
    mgpu::cudaFreeHost(host_ptr);
}

void test_matrixMul() {
    const int block_size = 16;
    const int wA = 5 * 2 * block_size, hA = wA;
    const int wB = 5 * 4 * block_size, hB = wA;
    void * h_A = mgpu::cudaMallocHost(sizeof(float) * wA * hA);
    void * h_B = mgpu::cudaMallocHost(sizeof(float ) * wB * hB);
    for(int i = 0; i < wA * hA; i++)
        *((float*)h_A + i) = 1.0f;
    for(int i = 0; i < wB * hB; i++)
        *((float*)h_B + i) = 0.1f;
    void * matA = mgpu::cudaMalloc(sizeof(float ) * hA * wA);
    void * matB = mgpu::cudaMalloc(sizeof(float ) * hB * wB);
    void * matC = mgpu::cudaMalloc(sizeof(float)  * hA * wB);
    mgpu::cudaMemcpy(matA, h_A, sizeof(float) * wA * hA, cudaMemcpyHostToDevice);
    mgpu::cudaMemcpy(matB, h_B, sizeof(float) * wB * hB, cudaMemcpyHostToDevice);
    mgpu::cudaFreeHost(h_A);
    mgpu::cudaFreeHost(h_B);
    dim3 threads(block_size, block_size, 1);
    dim3 grid(wB / threads.x, hA / threads.y, 1);
    mgpu::stream_t stream;
    mgpu::cudaStreamCreate(&stream, 1);
    mgpu::cudaLaunchKernel({grid, threads, 0, stream}, "/opt/custom/ptx/matrixMul.ptx", "matrixMul", matC, matA, matB, wA, wB);
    mgpu::cudaStreamSynchronize(stream);
    void * h_C = mgpu::cudaMallocHost(sizeof(float) * hA * wB);
    mgpu::cudaMemcpy(h_C, matC, sizeof(float) * hA * wB, cudaMemcpyDeviceToHost);
    mgpu::cudaFree(matA);
    mgpu::cudaFree(matB);
    mgpu::cudaFree(matC);
    printf("%f\n", *(float*)h_C);
    mgpu::cudaFreeHost(h_C);
}

int main() {
    test_vecAdd();
    test_matrixMul();
    //test_sm();
    return 0;
}
