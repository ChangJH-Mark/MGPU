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

// Return统一内存的host端地址
char* UnifyPointer::hostAddr() {
    return cpu_address;
}

// Return统一内存的device端地址
char* UnifyPointer::deviceAddr() {
    return gpu_address;
}

// Return内存大小
size_t UnifyPointer::len() {
    return size;
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