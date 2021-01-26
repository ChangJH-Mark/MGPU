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

typedef struct {
    UnifyPointer uptr;
    Error err;
}Mem;

Mem MemAlloc(size_t, MemType, char init_value = 0);

#endif