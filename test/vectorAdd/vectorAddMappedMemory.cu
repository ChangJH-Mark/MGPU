#include <cuda_runtime.h>
#include "allocator.h"
#define N 50000

__global__ void vectorAdd(float *a, float *b, float* c, int size) {
    int strip = blockDim.x;
    for(int i = threadIdx.x; i< size; i += strip) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    err_t err;
    auto h_a = MemAlloc(N, CPUPINMAP, &err);
    auto h_b = MemAlloc(N, CPUPINMAP, &err);
    auto h_c = MemAlloc(N, CPUPINMAP, &err);
    vectorAdd<<<1, 256>>>(h_a.hostAddr<float>(), h_b.hostAddr<float>(), h_c.hostAddr<float>(), N);
    h_a.free();
    h_b.free();
    h_c.free();
    return 0;
}