//
// Created by root on 2021/3/16.
//
#include <iostream>
#include <unistd.h>
#include <chrono>
#include <assert.h>
#include "client/api.h"

using namespace std;

void test_sm() {
    uint block_num = 10;
    uint size = sizeof(block_num) * block_num;
    void *dev_ptr1 = mgpu::cudaMalloc(size);
    void *host_ptr1 = mgpu::cudaMallocHost(size);
    mgpu::stream_t stream;
    mgpu::cudaStreamCreate(&stream);
    mgpu::cudaLaunchKernel({{block_num}, {1}, 0, stream}, "/opt/custom/ptx/specify_sm.ptx", "sm_ids", dev_ptr1);
    mgpu::cudaStreamSynchronize(stream);
    mgpu::cudaMemcpy(host_ptr1, dev_ptr1, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < block_num; i++) {
        printf("%d ", *((int *) (host_ptr1) + i));
    }
    printf("\n");
    mgpu::cudaFree(dev_ptr1);
    mgpu::cudaFreeHost(host_ptr1);
}

void test_vectorAdd() {
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
    printf("0x%x\n", *((int *) host_ptr + N / sizeof(int) - 1));
    mgpu::cudaFree(dev_ptr1);
    mgpu::cudaFree(dev_ptr2);
    mgpu::cudaFreeHost(host_ptr);
}

void test_matrixMul() {
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
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++)
            cout << static_cast<float *>(h_C)[i * block_size + j] << "\t";
        cout << endl;
    }
    mgpu::cudaFreeHost(h_C);
}

void test_multiGPUmatrixMul() {
    const int block_size = 16;
    const int wA = block_size, hA = wA;
    const int wB = block_size, hB = wA;
    void *h_A = mgpu::cudaMallocHost(sizeof(float) * wA * hA);
    void *h_B = mgpu::cudaMallocHost(sizeof(float) * wB * hB);
    // initial matrix
    for (int i = 0; i < wA * hA; i++)
        static_cast<float *>(h_A)[i] = 1.0f;
    for (int i = 0; i < wB * hB; i++)
        static_cast<float *>(h_B)[i] = 0.1f;
    dim3 threads(block_size, block_size);
    dim3 grid(wB / threads.x, hA / threads.y);
    auto res = mgpu::matrixMul_MGPU({h_A, wA, hA}, {h_B, wB, hB}, {grid, threads});
    auto value = static_cast<float *>(res.get());
    if (!value)
        exit(1);
    for (int i = 0; i < hA; i++) {
        for (int j = 0; j < wB; j++) {
            cout << value[i * wA + j] << " ";
        }
        cout << endl;
    }
    mgpu::cudaFreeHost(h_A);
    mgpu::cudaFreeHost(h_B);
}

void test_multiGPU() {
    const int N = 1 << 24;
    const int TaskNum = 6;
    mgpu::Task tasks[TaskNum];
    char *h_ptr_1, *h_ptr_2;
    for (auto & task : tasks) {
        task.hdn = 2;
        task.hds[0] = N;
        task.hds[1] = N;

        h_ptr_1 = (char *)mgpu::cudaMallocHost(N);
        h_ptr_2 = (char *)mgpu::cudaMallocHost(N);
        memset(h_ptr_1, 0x1, N);
        memset(h_ptr_2, 0x2, N);
        task.p_size = fillParameters(task.param, 0, h_ptr_1, h_ptr_2, (void *)NULL, N / sizeof(int));

        task.dn = 1;
        task.dev_alloc_size[0] = N;
        strcpy(task.ptx, "/opt/custom/ptx/vectorAdd.cubin");
        strcpy(task.kernel, "vectorAdd");
        task.conf = {.grid = {(N/sizeof(int) / 256) + 1},.block= 256};
    }
    auto ret = mgpu::MulTaskMulGPU(TaskNum, tasks);
    for(int i = 0;i<TaskNum;i++) {
        cout << "task " << i << " success is " << ret.success[i] << " dev: " << ret.bind_dev[i] << endl;
        void *result = (void *)*(unsigned long long *)(ret.msg[i] + 2 * sizeof(void *));
        cout << "start check result ptr " << hex << result << dec << endl;
        mgpu::cudaMemcpy(h_ptr_1, result, N, cudaMemcpyDeviceToHost);
        for(int j = 0; j< N;j++) {
            if(h_ptr_1[j] != 0x03)
                cout << "for task " << i << " result " << j << " is not 0x03 but " << hex << (int)h_ptr_1[j] << dec << endl;
        }
    }
}

int main() {
    test_vectorAdd();
    test_matrixMul();
//    test_multiGPU();
//    test_multiGPUmatrixMul();
    //test_sm();
    return 0;
}
