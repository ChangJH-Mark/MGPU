#ifndef ALLOCATOR_H_
#define ALLOCATOR_H_

#include <cuda.h>
#include <stddef.h>
#include "unify_pointer.h"
#include "error.h"

#ifndef MEMTYPE_T_
#define MEMTYPE_T_
enum MemType {
    GPUMEM,
    CPUNOPINNOMAP,
    CPUNOPINMAP,
    CPUPINNOMAP,
    CPUPINMAP,
};
#endif

// Return统一指针类型
// 输入内存大小size_t byte
//          内存类型如GPUMEM、CPUPINMAP
//          err_t的指针，向client返回错误码
//          内存初始值init_value，默认是0
UnifyPointer MemAlloc(size_t, MemType, err_t* err, char init_value = 0);

#endif