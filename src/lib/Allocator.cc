#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include "allocator.h"

UnifyPointer MemAlloc(size_t size, MemType type, err_t *err, char init_value /*default 0*/)
{
    if (err == nullptr)
    {
        *err = UM_ERR_ARG_EMPTY;
        return UnifyPointer();
    }
    *err = UM_SUCCESS;
    switch (type)
    {
    case GPUMEM:
    {
        void *dev_p = nullptr;
        *err = cudaMalloc(&dev_p, size);
        if (*err != UM_SUCCESS)
        {
            return UnifyPointer();
        }
        auto res = UnifyPointer((char *)dev_p, nullptr, size, GPUMEM);
        return res;
    }
    case CPUNOPINNOMAP:
    {
        void *address = malloc(size);
        memset(address, init_value, size);
        auto res = UnifyPointer(nullptr, (char *)address, size, type);
        return res;
        break;
    }
    case CPUPINNOMAP:
    {
        char *cpu, *gpu;
        *err = cudaMallocHost(&cpu, size);
        if (*err != UM_SUCCESS)
        {
            return UnifyPointer();
        }
        auto res = UnifyPointer(nullptr, cpu, size, CPUPINNOMAP);
        return res;
        break;
    }
    case CPUPINMAP:
    {
        char *cpu, *gpu;
        *err = cudaHostAlloc(&cpu, size, cudaHostAllocMapped);
        if (*err != CUDA_SUCCESS)
        {
            return UnifyPointer();
        }
#ifdef CUDA_VERSION
        gpu = cpu;
        if (CUDA_VERSION <= 4000)
        {
            *err = cudaHostGetDevicePointer(&gpu, cpu, 0);
            if (*err != UM_SUCCESS)
                return UnifyPointer();
        }
#endif
        auto res = UnifyPointer(cpu, gpu, size, CPUPINMAP);
        return res;
        break;
    }
    default:
    {
        *err = UM_BADFLAG;
        return UnifyPointer();
    }
    } // switch
}