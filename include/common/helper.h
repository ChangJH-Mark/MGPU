//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_HELPER_H
#define FASTGPU_HELPER_H
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
using namespace std;

#define cudaCheck(x) if(x!=cudaSuccess) { \
cerr << "cuda error: " << x << " message: " << cudaGetErrorString(x) << endl; \
exit(EXIT_FAILURE);\
}
#endif //FASTGPU_HELPER_H
