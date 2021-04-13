//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_HELPER_H
#define FASTGPU_HELPER_H
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>
using namespace std;

#define cudaCheck(x) if((x)!=cudaSuccess) { \
cerr << __FUNCTION__ <<" cuda error: " << x << " message: " << cudaGetErrorString(x) << endl; \
exit(EXIT_FAILURE);\
}

template<typename T>
unsigned fillParameters(char *buff, unsigned int offset, T value) {
    offset = (offset + __alignof(value) -1) & ~(__alignof(value) -1);
    memcpy((buff + offset), &value, sizeof(value));
    offset += sizeof(value);
    return offset;
}

template<typename T, typename... Args>
unsigned fillParameters(char *buff, unsigned offset, T value, Args... args)
{
    offset = (offset + __alignof(value) -1) & ~(__alignof(value) -1);
    memcpy((buff + offset), &value, sizeof(value));
    offset += sizeof(value);
    return fillParameters(buff, offset, args...);
}
#endif //FASTGPU_HELPER_H
