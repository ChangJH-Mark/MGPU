//
// Created by mark on 2021/3/2.
//

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <sys/mman.h>
#define DEFAULT_PINNED_GENERIC_MEMORY true
// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

using namespace std;
__global__ void init_array(int *g_data, int *factor, int num_iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i=0; i<num_iterations; i++)
    {
        g_data[idx] += *factor;    // non-coalesced on purpose, to burn time
    }
}

bool correct_data(int *a, const int n, const int c)
{
    for (int i = 0; i < n; i++)
    {
        if (a[i] != c)
        {
            printf("%d: %d %d\n", i, a[i], c);
            return false;
        }
    }

    return true;
}

inline void AllocateHostMemory(bool bPinGenericMemory, int **pp_a, int **ppAligned_a, int nbytes) {
#if !defined(__arm__) && !defined(__aarch64__)
    if (bPinGenericMemory) {
        // allocate a generic page-aligned chunck of system memory
        printf("> mmap() allocating %4.2f Mbytes (generic page-aligned system memory)\n", (float)nbytes/1048576.0f);
        *pp_a = (int *) mmap(NULL, (nbytes + MEMORY_ALIGNMENT), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, 0);

        *ppAligned_a = (int *)ALIGN_UP(*pp_a, MEMORY_ALIGNMENT);

        cudaHostRegister(*ppAligned_a, nbytes, cudaHostRegisterMapped);
    }else
#endif
    {
        printf("> cudaMallocHost() allocating %4.2f Mbytes of system memory\n", (float)nbytes/1048576.0f);
        // allocate host memory (pinned is required for achieve asynchronicity)
        cudaMallocHost((void **)pp_a, nbytes);
        *ppAligned_a = *pp_a;
    }
}

inline void
FreeHostMemory(bool bPinGenericMemory, int **pp_a, int **ppAligned_a, int nbytes)
{
#if CUDART_VERSION >= 4000
#if !defined(__arm__) && !defined(__aarch64__)
    // CUDA 4.0 support pinning of generic host memory
    if (bPinGenericMemory)
    {
        // unpin and delete host memory
        cudaHostUnregister(*ppAligned_a);
#ifdef WIN32
        VirtualFree(*pp_a, 0, MEM_RELEASE);
#else
        munmap(*pp_a, nbytes);
#endif
    }
    else
#endif
#endif
    {
        cudaFreeHost(*pp_a);
    }
}

int main() {
    int nstreams = 4; // 多少个streams
    int nreps = 10; // 重复多少次
    int n = 16 * 1024 * 1024; // 元素规模
    int nbytes = n * sizeof(int);
    dim3 threads, blocks;
    float elapsed_time, time_memcpy, time_kernel;
    float scale_factor = 1.0f;

    bool bPinGenericMemory = DEFAULT_PINNED_GENERIC_MEMORY;
    int device_sync_method = cudaDeviceBlockingSync;

    int niterations = 5; // 执行多少次
    int cuda_device = 0;
    cudaSetDevice(cuda_device);
    if(bPinGenericMemory) {
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, cuda_device);
        printf("Device: <%s> canMapHostMemory: %s\n", deviceProp.name, deviceProp.canMapHostMemory ? "Yes" : "No");
        if(deviceProp.canMapHostMemory == 0) {
            printf("Using cudaMallocHost, CUDA device does not support mapping of generic host memory\n");
            bPinGenericMemory = false;
        }
    }

    scale_factor = max(32.0f / (128 * 4), 1.0f);
    n = (int)rint(float(n) / scale_factor);
    printf("> scale_factor = %1.4f\n", 1.0f/scale_factor);
    printf("> array_size   = %d\n\n", n);
    cudaSetDeviceFlags(device_sync_method | (bPinGenericMemory ? cudaDeviceMapHost : 0));

    // allocate host memory
    int c = 5;
    int *h_a = nullptr;
    int *hAligned_a = nullptr;
    // allocate Host memory
    AllocateHostMemory(bPinGenericMemory, &h_a, &hAligned_a, nbytes);

    // allocate device memory
    int *d_a, *d_c; // data pointer & init value
    cudaMalloc((void**)&d_a, nbytes);
    cudaMemset(d_a, 0x0, nbytes);
    cudaMalloc((void**)&d_c, sizeof(int));
    cudaMemcpy(d_c, &c, sizeof(int), cudaMemcpyHostToDevice);

    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
    for (int i = 0; i < nstreams; i++ ) {
        cudaStreamCreate(&streams[i]);
    }

    // create CUDA event handles
    // use blocking sync
    cudaEvent_t start_event, stop_event;
    int eventflags = ((device_sync_method == cudaDeviceBlockingSync) ? cudaEventBlockingSync: cudaEventDefault);

    cudaEventCreateWithFlags(&start_event, eventflags);
    cudaEventCreateWithFlags(&stop_event, eventflags);

    // time memcopy from device
    cudaEventRecord(start_event, 0);
    cudaMemcpyAsync(hAligned_a, d_a, nbytes, cudaMemcpyDeviceToHost, streams[0]);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event); // block until the event is actually recorded
    cudaEventElapsedTime(&time_memcpy, start_event, stop_event);

    printf("memcopy:\t%.2f\n", time_memcpy);

    // time kernel
    threads = dim3(512, 1);
    blocks = dim3( n / threads.x, 1);
    cudaEventRecord(start_event, 0);
    init_array<<<blocks, threads, 0, streams[0]>>>(d_a, d_c, niterations);
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&time_kernel, start_event, stop_event);
    printf("kernel:\t\t%.2f\n", time_kernel);

    //////////////////////////////////////////////////////////////////////
    // time non-streamed execution for reference
    threads = dim3(512, 1);
    blocks = dim3(n / threads.x, 1);
    cudaEventRecord(start_event, 0);
    for (int k = 0; k < nreps; k++)
    {
        init_array<<<blocks, threads>>>(d_a, d_c, niterations);
        cudaMemcpy(hAligned_a, d_a, nbytes, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
    printf("non-streamed:\t%.2f\n", elapsed_time / nreps);

    //////////////////////////////////////////////////////////////////////
    // time execution with nstreams streams
    threads=dim3(512,1);
    blocks=dim3(n/(nstreams*threads.x),1);
    memset(hAligned_a, 255, nbytes); // set host memory to 1s, for test correctness
    cudaMemset(d_a, 0, nbytes); // set device to all 0s for test correctness
    cudaEventRecord(start_event, 0);

    for(int k = 0; k< nreps; k++)
    {
        for(int i = 0; i < nstreams; i++) {
            init_array<<<blocks, threads, 0, streams[i]>>>(d_a + i * n / nstreams, d_c, niterations);
        }
        // asynchronously launch nstreams memcopies.  Note that memcopy in stream x will only
        //   commence executing when all previous CUDA calls in stream x have completed
        for(int i = 0; i < nstreams; i ++){
            cudaMemcpyAsync(hAligned_a + i * n / nstreams, d_a + i*n / nstreams, nbytes/ nstreams, cudaMemcpyDeviceToHost, streams[i]);
        }
    }
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
    printf("%d streams:\t%.2f\n", nstreams, elapsed_time / nreps);

    // check whether the output is correct
    printf("-------------------------------\n");
    bool bResults = correct_data(hAligned_a, n, c*nreps*niterations);

    // release resources
    for(int i = 0; i< nstreams; i++){
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    // Free cudaMallocHost or Generic Host allocated memory (from CUDA 4.0)
    FreeHostMemory(bPinGenericMemory, &h_a, &hAligned_a, nbytes);

    cudaFree(d_a);
    cudaFree(d_c);

    return bResults ? EXIT_SUCCESS : EXIT_FAILURE;
}