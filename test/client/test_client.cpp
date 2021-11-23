//
// Created by root on 2021/3/16.
//
#include <iostream>
#include <unistd.h>
#include <chrono>
#include <assert.h>
#include <atomic>
#include <cuda.h>
#include "client/api.h"

using namespace std;

void test_vectorAdd()
{
    const int N = 1 << 24;
    void *dev_ptr1 = mgpu::cudaMalloc(N);
    void *dev_ptr2 = mgpu::cudaMalloc(N);
    mgpu::cudaMemset(dev_ptr1, 0x1, N);
    mgpu::cudaMemset(dev_ptr2, 0x2, N);
    void *host_ptr = mgpu::cudaMallocHost(N);
    printf("dev_ptr1: 0x%lx dev_ptr2: 0x%lx, host_ptr: 0x%lx\n", dev_ptr1, dev_ptr2, host_ptr);
    mgpu::cudaLaunchKernel({{1 << 14}, {256}, 0, NULL}, "/opt/custom/ptx/vectorAdd.cubin", "vectorAdd", dev_ptr1,
                           dev_ptr2, dev_ptr2,
                           int(N / sizeof(int)));
    mgpu::cudaMemcpy(host_ptr, dev_ptr2, N, cudaMemcpyDeviceToHost);
    //    for(int i = 0; i < N/sizeof(int); i++) {
    //        printf("%d: 0x%x\n", i, ((int *)(host_ptr))[i]);
    //    }
    printf("0x%x\n", *((int *)host_ptr + N / sizeof(int) - 1));
    mgpu::cudaFree(dev_ptr1);
    mgpu::cudaFree(dev_ptr2);
    mgpu::cudaFreeHost(host_ptr);
}

chrono::steady_clock::time_point s;
chrono::steady_clock::time_point e;

atomic<int> cnts(0);

void vectorAdd(char *host_ptr1, char *host_ptr2, int dev)
{
    ::cudaSetDevice(dev);
    const int N = 1 << 24;
    void *dev_ptr1, *dev_ptr2, *dev_ptr3;

    memset(host_ptr1, 0x1, N);
    memset(host_ptr2, 0x2, N);

    cudaCheck(cudaMalloc(&dev_ptr1, N));
    cudaCheck(cudaMalloc(&dev_ptr2, N));
    cudaCheck(cudaMalloc(&dev_ptr3, N));
    cudaCheck(cudaMemcpy(dev_ptr1, host_ptr1, N, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dev_ptr2, host_ptr2, N, cudaMemcpyHostToDevice));

    void *host_ptr;
    CUmodule mod;
    CUfunction func;
    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream));
    cudaCheck(cuModuleLoad(&mod, "/opt/custom/ptx/vectorAdd.cubin"));
    cudaCheck(cuModuleGetFunction(&func, mod, "vectorAdd"));
    char param[1024];
    int size = fillParameters(param, 0, dev_ptr1, dev_ptr2, dev_ptr3, N / sizeof(int));
    void *extra[] = {
            CU_LAUNCH_PARAM_BUFFER_POINTER, param,
            CU_LAUNCH_PARAM_BUFFER_SIZE, &size,
            CU_LAUNCH_PARAM_END};
    dim3 grid = (N / 1024);
    dim3 block = 256;
    cudaCheck(cuLaunchKernel(func, grid.x, grid.y, grid.z, block.x, block.y, block.z, 0, stream, NULL, extra));
    cudaCheck(cudaStreamSynchronize(stream));
    cudaCheck(cudaMemcpy(host_ptr2, dev_ptr3, N, cudaMemcpyDeviceToHost));
    // cudaCheck(cudaFree(dev_ptr1));
    // cudaCheck(cudaFree(dev_ptr2));
}

void test_matrixMul()
{
    const int block_size = 16;
    const int wA = 5 * 2 * block_size, hA = wA;
    const int wB = 5 * 4 * block_size, hB = wA;
    void *h_A = mgpu::cudaMallocHost(sizeof(float) * wA * hA);
    void *h_B = mgpu::cudaMallocHost(sizeof(float) * wB * hB);
    for (int i = 0; i < wA * hA; i++)
        static_cast<float *>(h_A)[i] = 1.0f;
    for (int i = 0; i < wB * hB; i++)
        static_cast<float *>(h_B)[i] = 0.1f;
    void *matA = mgpu::cudaMalloc(sizeof(float) * hA * wA);
    void *matB = mgpu::cudaMalloc(sizeof(float) * hB * wB);
    void *matC = mgpu::cudaMalloc(sizeof(float) * hA * wB);
    mgpu::cudaMemcpy(matA, h_A, sizeof(float) * wA * hA, cudaMemcpyHostToDevice);
    mgpu::cudaMemcpy(matB, h_B, sizeof(float) * wB * hB, cudaMemcpyHostToDevice);
    cout << hex << "h_A: " << h_A << " h_B: " << h_B << " matA: " << matA << " matB: " << matB << " matC: " << matC
         << endl;
    mgpu::cudaFreeHost(h_A);
    mgpu::cudaFreeHost(h_B);
    dim3 threads(block_size, block_size, 1);
    dim3 grid(wB / threads.x, hA / threads.y, 1);
    mgpu::cudaLaunchKernel({grid, threads, 0, NULL}, "/opt/custom/ptx/matrixMul.cubin", "MatrixMulCUDA", matC, matA,
                           matB,
                           wA, wB);
    void *h_C = mgpu::cudaMallocHost(sizeof(float) * hA * wB);
    cout << "h_C: " << h_C << endl;
    mgpu::cudaMemcpy(h_C, matC, sizeof(float) * hA * wB, cudaMemcpyDeviceToHost);
    mgpu::cudaFree(matA);
    mgpu::cudaFree(matB);
    mgpu::cudaFree(matC);
    for (int i = 0; i < block_size; i++)
    {
        for (int j = 0; j < block_size; j++)
            cout << static_cast<float *>(h_C)[i * block_size + j] << "\t";
        cout << endl;
    }
    mgpu::cudaFreeHost(h_C);
}

void test_multiGPU()
{
    const int N = 1 << 24;
    const int TaskNum = 6;
    mgpu::Task tasks[TaskNum];
    vector<void *> ptrs;
    char *h_ptr_1, *h_ptr_2;
    for (auto &task : tasks)
    {
        task.hdn = 2;
        task.hds[0] = N;
        task.hds[1] = N;

        h_ptr_1 = (char *)mgpu::cudaMallocHost(N);
        h_ptr_2 = (char *)mgpu::cudaMallocHost(N);
        ptrs.push_back(h_ptr_1);
        ptrs.push_back(h_ptr_2);
        memset(h_ptr_1, 0x1, N);
        memset(h_ptr_2, 0x2, N);
        task.p_size = fillParameters(task.param, 0, h_ptr_1, h_ptr_2, (void *)NULL, N / sizeof(int));

        task.dn = 1;
        task.dev_alloc_size[0] = N;
        strcpy(task.ptx, "/opt/custom/ptx/vectorAdd.cubin");
        strcpy(task.kernel, "vectorAdd");
        task.conf = {.grid = {(N / sizeof(int) / 256) + 1}, .block = 256};
    }
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    auto ret = mgpu::MulTaskMulGPU(TaskNum, tasks);
    for (int i = 0; i < TaskNum; i++)
    {
        void *result = (void *)*(unsigned long long *)(ret.msg[i] + 2 * sizeof(void *));
        mgpu::cudaMemcpy(h_ptr_1, result, N, cudaMemcpyDeviceToHost);
    }
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << " total cost " << chrono::duration_cast<chrono::microseconds>(end - start).count() << endl;
    for (int i = 0; i < TaskNum; i++)
    {
        cout << "task" << i << " chose " << ret.bind_dev[i] << endl;
    }
    for (auto p : ptrs)
    {
        mgpu::cudaFreeHost(p);
    }
}

int main()
{
    struct example
    {
        char *host_ptr1, *host_ptr2;
        int dev;
    } E[6];
    int id = 0;
    srand(time(NULL));
    for (int i = 0; i < 6; i++)
    {
        id = 0;
        cudaSetDevice(id);
        cudaFree(0);
        cudaCheck(cudaMallocHost(&(E[i].host_ptr1), 1 << 24));
        cudaCheck(cudaMallocHost(&(E[i].host_ptr2), 1 << 24));
        E[i].dev = id;
        cout << id << endl;
    }
    thread ts[6];
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    for (int i = 0; i < 6; i++)
    {
        ts[i] = std::move(thread(vectorAdd, E[i].host_ptr1, E[i].host_ptr2, E[i].dev));
    }
    for (int i = 0; i < 6; i++)
        ts[i].join();
    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << " total cost " << chrono::duration_cast<chrono::microseconds>(end - start).count() << endl;
    // test_multiGPU();
    return 0;
}
