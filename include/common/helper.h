//
// Created by root on 2021/3/12.
//

#ifndef FASTGPU_HELPER_H
#define FASTGPU_HELPER_H
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>
#include "message.h"
using namespace std;

#define cudaCheck(x) { \
       cudaError_t err = static_cast<cudaError_t>(x); \
       if(err != cudaSuccess) { \
          cerr << __FILE__ << " " << __FUNCTION__ << " " << __LINE__ << " cuda error: " << err << " message: " << cudaGetErrorString(err) << endl; \
          exit(EXIT_FAILURE);\
       }\
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

template<typename T, typename... Args>
void initTask(mgpu::Task* t, uint hdn, uint dn,const vector<void *>& hda, const vector<size_t>& hds, const vector<size_t>& ds, mgpu::LaunchConf conf, const char* name, const char* kernel, Args... args) {
    t->hdn = hdn;
    t->dn = dn;
    t->conf = conf;
    strcpy(t->kernel, kernel);
    strcpy(t->ptx, name);
    t->p_size = 0;
    for(int i =0;i<hdn;i++) {
        t->hds[i] = hds[i];
        t->p_size = fillParameters(t->param, t->p_size, hda[i]);
    }
    for(int i = 0;i<dn;i++)
    {
        t->dev_alloc_size[i] = ds[i];
        t->p_size = fillParameters(t->param, t->p_size, nullptr);
    }
    t->p_size = fillParameters(t->param, t->p_size, args...);
}
#endif //FASTGPU_HELPER_H
