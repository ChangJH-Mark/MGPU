//
// Created by root on 2021/4/2.
//

__device__ uint get_smid() {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

extern "C" __global__ void sm_ids(uint* store) {
    if(threadIdx.x == 0){
        store[blockIdx.x] = get_smid();
    }
}