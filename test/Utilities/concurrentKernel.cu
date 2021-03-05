//
// Created by mark on 2021/3/2.
//

#include <cuda_runtime.h>
#include <iostream>
using namespace std;

int main() {
    int nKernels = 8;
    int nStreams = nKernels + 1;
    int nbytes = nKernels * sizeof(clock_t);
    float kernel_time = 10;
    float elapsed_time;
    cudaSetDevice(0);
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    if(deviceProp.concurrentKernels != 0) {
        cout << "GPU does not support concurrent kernels execution" << endl;
        cout << "CUDA kernel runs will be serialized\n" << endl;
    }
    cout << deviceProp.asyncEngineCount << endl;
    clock_t *a = nullptr;
    cudaMallocHost((void **)&a, nbytes);
    clock_t *d_a = nullptr;
    cudaMalloc((void **)&d_a, nbytes);

    auto streams = (cudaStream_t *)malloc(nStreams * sizeof(cudaStream_t));

    for(int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, streams[0]);
    cudaMemcpyAsync(d_a, a, nbytes, cudaMemcpyHostToDevice, streams[1]);
}