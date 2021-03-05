//
// Created by root on 2021/3/3.
//
#include <stdio.h>
#define DATA_SIZE 10000000

__global__ void vecAdd(int *a, int *b, int *c, int num) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int skip = gridDim.x * blockDim.x;
    for(int i = id; i < num; i+= skip) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vecAbstract(int *a, int *b, int *c, int num) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int skip = gridDim.x * blockDim.x;
    for(int i = id; i < num; i+= skip) {
        c[i] = a[i] - b[i];
    }
}

void correctCheck(int *h, int type) {
    switch (type) {
        case 0:{
            for(int i = 0; i < DATA_SIZE; i ++)
            {
                if(h[i] != 0x1010101)
                {
                    printf("abstract fail at %d value: %d", i, h[i]);
                    return;
                }
            }
        }
        break;
        case 1: {
            for(int i = 0; i < DATA_SIZE; i ++)
            {
                if(h[i] != 0x3030303)
                {
                    printf("add fail at %d value: %d", i, h[i]);
                    return;
                }
            }
        }
    }
}

int main() {
    int nstreams = 2;
    cudaSetDevice(0);
    cudaSetDeviceFlags(cudaDeviceBlockingSync | cudaDeviceMapHost);
    auto streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
    for(int i = 0; i < nstreams; i++ ){
        cudaStreamCreate(&streams[i]);
    }

    cudaEvent_t start_event, stop_event;
    cudaEventCreateWithFlags(&start_event, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&stop_event, cudaEventBlockingSync);

    float time;
    int * d_a, *d_b, *d_c_add, *d_c_abstract;
    int * h_a;
    // allocate memory
    cudaMallocHost(&h_a, DATA_SIZE * sizeof(int));
    cudaMalloc(&d_a, DATA_SIZE * sizeof(int));
    cudaMalloc(&d_b, DATA_SIZE * sizeof(int));
    cudaMalloc(&d_c_add, DATA_SIZE * sizeof(int));
    cudaMalloc(&d_c_abstract, DATA_SIZE * sizeof(int));
    cudaMemset(d_a, 0x2, DATA_SIZE * sizeof(int));
    cudaMemset(d_b, 0x1, DATA_SIZE * sizeof(int));

    dim3 blockDim(512);
    dim3 gridDim(1);
    cudaEventRecord(start_event, 0);
    vecAbstract<<<gridDim, blockDim, 0, streams[0]>>>(d_a, d_b, d_c_abstract, DATA_SIZE);
    vecAdd<<<gridDim, blockDim, 0, streams[1]>>>(d_a, d_b, d_c_add, DATA_SIZE);

    cudaMemcpyAsync(h_a, d_c_abstract, sizeof(int) * DATA_SIZE, cudaMemcpyDeviceToHost, streams[0]);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time, start_event, stop_event);
    printf("vecabstract cost: %.2f \n", time);
    cudaStreamSynchronize(streams[0]);
    correctCheck(h_a, 0);
    cudaMemcpyAsync(h_a, d_c_add, sizeof(int) * DATA_SIZE, cudaMemcpyDeviceToHost, streams[1]);
    cudaDeviceSynchronize();
    correctCheck(h_a, 1);
}