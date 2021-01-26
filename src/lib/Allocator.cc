#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include "allocator.h"

Mem MemAlloc(size_t size, MemType type, char init_value /*default 0*/)
{
    Mem res;
    switch (type)
    {
    case CPUNOPINNOMAP:
    {
        void *address = malloc(size);
        memset(address, init_value, size);
        res.uptr = UnifyPointer(nullptr, (char *)address, size, type);
        res.err = UM_SUCCESS;
        return res;
        break;
    }
    case CPUNOPINMAP:
    {
        return res;
        break;
    }
    case CPUPINNOMAP:
    {
        char *cpu, *gpu;
        res.err = cudaMallocHost(&cpu, size);
        if (res.err != UM_SUCCESS)
        {
            return res;
        }
        res.uptr = UnifyPointer(nullptr, cpu, size, CPUPINNOMAP);
        break;
    }
    case CPUPINMAP:
    {
        char *cpu, *gpu;
        auto err = cudaHostAlloc(&cpu, size, cudaHostAllocMapped);
        if (err != CUDA_SUCCESS)
        {
            res.err = err;
            return res;
        }
        res.err = err;
#ifdef CUDA_VERSION
        gpu = cpu;
        if (CUDA_VERSION <= 4000)
        {
            res.err = cudaHostGetDevicePointer(&gpu, cpu, 0);
            if (res.err != UM_SUCCESS)
                return res;
        }
#endif
        res.uptr = UnifyPointer(cpu, gpu, size, CPUPINMAP);
        return res;
        break;
    }
    default:
    {
        res.err = UM_BADFLAG;
        return res;
    }
    } // switch
}