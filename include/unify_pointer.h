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
    CPUPINNOMAP,
    CPUPINMAP,
};
#endif


// 将cpu端内存、gpu端内存、统一管理起来
class UnifyPointer
{
private:
    void *cpu_address;
    void *gpu_address;
    size_t size;
    MemType mem_type;
public:
    UnifyPointer(){};
    UnifyPointer(void *gpuaddr, void *cpuaddr, size_t size, MemType type);
    template<class T> T* hostAddr() {return (T*)cpu_address;}
    template<class T> T* deviceAddr() {return (T*)gpu_address;}
    size_t len() {return size;};
    err_t free();
};

#endif