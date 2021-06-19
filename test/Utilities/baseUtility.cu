//
// Created by mark on 2021/3/2.
//

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <assert.h>

#define BLOCK_SIZE 256
#define N  128 * 1024 * BLOCK_SIZE
#define ITERS 10

using namespace std;

inline void CHECK(cudaError_t x) {
    if(x != cudaSuccess)
    {
        cout << "error: " << cudaGetErrorString(x) << endl;
        exit(EXIT_FAILURE);
    }
}

#define SMID_MASK 0xf
#define ITERS_MASK 0xf0
#define WORKER_MASK 0xff00

#define GET_SID(flags) (flags & SMID_MASK)
#define GET_WID(flags) (( flags & WORKER_MASK) >> 8)
#define SET_WID(flags, worker) (flags = (flags & ~(WORKER_MASK)) + ((worker) << 8))
#define GET_ITERS(flags) ((flags & ITERS_MASK) >> 4)
#define SET_ITERS(flags, times) (flags = (flags & ~(ITERS_MASK)) + ((times) << 4))

__device__ int sm_low;
__device__ int sm_high;
__device__ int counts;
__device__ int blim;
__device__ int bcnts[6];

__global__ void vecAdd(int *a, int *b, int n) {
    __shared__ int start_block;
    __shared__ int flags;
    if (threadIdx.x == 0) {
        int worker;
        asm("mov.u32 %0, %smid;" : "=r"(flags));
        if(GET_SID(flags) < sm_low || GET_SID(flags) > sm_high) {
            start_block = -1;
        }
        else if((worker = atomicAdd(bcnts + GET_SID(flags), 1)) >= blim) {
            start_block = -1;
        }
        else {
            SET_WID(flags, worker);
        }
    }
    __syncthreads();
    if(start_block == -1)
        return;
    while(start_block != -1) {
        if (threadIdx.x == 0) {
            // sm checks
            if (GET_SID(flags) < sm_low || GET_SID(flags) > sm_high) {
                start_block = -1;
            }
            // worker check
            else if (GET_WID(flags) >= blim) {
                start_block = -1;
            }
            else {
                start_block = atomicAdd(&counts, ITERS);
                if(start_block >= N / BLOCK_SIZE) {
                    start_block = -1;
                }
                else {
                    SET_ITERS(flags, min(start_block + ITERS, N / BLOCK_SIZE) - start_block);
                }
            }
        }
        __syncthreads();
        if (start_block == -1) {
            break;
        }
        for (int i = start_block; i < start_block + GET_ITERS(flags); i++) {
            int index = threadIdx.x + i * blockDim.x;
            if (index >= n)
                return;
            b[index] = a[index] + b[index];
            __syncthreads();
        }
    }
}

void run_kernel(int *dev_a, int *dev_b, int n) {
    cudaEvent_t sync;
    CHECK(cudaEventCreate(&sync));
    int sm = 0;
    CHECK(cudaMemcpyToSymbol(sm_low, &sm, sizeof(sm), 0, cudaMemcpyHostToDevice));
    sm = 6;
    CHECK(cudaMemcpyToSymbol(sm_high, &sm, sizeof(sm), 0, cudaMemcpyHostToDevice));
    int count = 0;
    CHECK(cudaMemcpyToSymbol(counts, &count, sizeof(count), 0, cudaMemcpyHostToDevice));
    int lim = 32;
    CHECK(cudaMemcpyToSymbol(blim, &lim, sizeof(lim), 0, cudaMemcpyHostToDevice));
    int blocks[6] = {0};
    CHECK(cudaMemcpyToSymbol(bcnts, blocks, sizeof(int) * 6, 0, cudaMemcpyHostToDevice));
    vecAdd<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(dev_a, dev_b, N);
    CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
//    while(count < N / BLOCK_SIZE) {
//        int blocks[6] = {0};
//        CHECK(cudaMemcpyToSymbol(bcnts, blocks, sizeof(int) * 6, 0, cudaMemcpyHostToDevice));
//        vecAdd<<<32 * 6, BLOCK_SIZE>>>(dev_a, dev_b, N);
//        CHECK(cudaGetLastError());
//        CHECK(cudaEventRecord(sync));
//        CHECK(cudaEventSynchronize(sync));
//        CHECK(cudaMemcpyFromSymbol(&count, counts, sizeof(count), 0, cudaMemcpyDeviceToHost));
//    }
}

int main() {
    int *dev_a, *dev_b;
    int err;
    err = cudaMalloc(&dev_a, sizeof(int) * N);
    err = cudaMalloc(&dev_b, sizeof(int) * N);

    err = cudaMemset(dev_a, 1, sizeof(int) * N);
    err = cudaMemset(dev_b, 2, sizeof(int) * N);

    auto start = chrono::steady_clock::now();
    run_kernel(dev_a, dev_b, N);
    cout << chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - start).count() << endl;

    int *res = static_cast<int *>(malloc(sizeof(int) * N));
    memset(res, 0, sizeof(int) * N);
    err = cudaMemcpy(res, dev_b, sizeof(int) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        if (res[i] != 0x03030303) {
            cout << "at : " << dec << i << " value is not 3, but : " << hex << res[i] << endl;
            break;
        }
    }
    cout << " test pass " << endl;

    cudaFree(dev_a);
    cudaFree(dev_b);
    free(res);
}
