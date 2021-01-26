#ifndef UNIFY_POINTER_H_
#define UNIFY_POINTER_H_

#include <stddef.h>
#include "error.h"

#ifndef MEMTYPE_T_
#define MEMTYPE_T_
enum MemType
{
    GPUMEM,
    CPUNOPINNOMAP,
    CPUNOPINMAP,
    CPUPINNOMAP,
    CPUPINMAP,
};
#endif

class UnifyPointer
{
public:
    char *cpu_address;
    char *gpu_address;
    size_t size;
    MemType mem_type;

public:
    UnifyPointer(){};
    UnifyPointer(char *gpuaddr, char *cpuaddr, size_t size, MemType type);
    Error free();
};

#endif