#include <cuda_runtime.h>
#include <stdlib.h>
#include "unify_pointer.h"

UnifyPointer::UnifyPointer(void *gpuaddr, void *cpuaddr, size_t size, MemType type)
{
    cpu_address = cpuaddr;
    gpu_address = gpuaddr;
    this->size = size;
    this->mem_type = type;
}

err_t UnifyPointer::free()
{
    switch (mem_type)
    {
    case GPUMEM:
    {
        return cudaFree(gpu_address);
    }
    case CPUPINMAP:
    case CPUPINNOMAP:
    {
        return cudaFree(cpu_address);
        break;
    }
    case CPUNOPINNOMAP:
    {
        std::free(cpu_address);
        return UM_SUCCESS;
    }
    default:
        break;
    }
}