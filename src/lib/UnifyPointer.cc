#include <cuda_runtime.h>
#include <stdlib.h>
#include "unify_pointer.h"

UnifyPointer::UnifyPointer(char *gpuaddr, char *cpuaddr, size_t size, MemType type)
{
    cpu_address = cpuaddr;
    gpu_address = gpuaddr;
    this->size = size;
    this->mem_type = type;
}

Error UnifyPointer::free()
{
    switch (mem_type)
    {
    case CPUPINMAP:
    case CPUPINNOMAP:
        return cudaFree(cpu_address);
        break;
    case CPUNOPINNOMAP:
        std::free(cpu_address);
    default:
        break;
    }
}