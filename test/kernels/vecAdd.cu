//
// Created by root on 2021/3/25.
//

extern "C" __global__ void vecAdd(int *a, int *b, int num) {
    for(int i= threadIdx.x; i< num; i+= blockDim.x){
        b[i] = a[i] + b[i];
    }
}