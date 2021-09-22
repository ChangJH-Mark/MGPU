//
// Created by root on 2021/3/25.
//
#define ITERS 10
__device__ void vecAdd(int *a, int *b, int num, uint3 blockIDX, dim3 gridDIM);

__device__ uint get_smid() {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ int finished = 0;

extern "C" __global__ void vecAddProxy(int *a, int *b, int num, int sm_low, int sm_high, dim3 grid, int blocks)
{
    // reside on sm (sm >= sm_low && sm < sm_high)
    bool leader = false;
    __shared__ bool terminate;
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        leader = true;
    }
    if(leader)
    {
        terminate = false;
        int sm_id = get_smid();
        if(sm_id < sm_low || sm_id >= sm_high) {
            terminate = true;
//            printf("parameter a %lx b %lx num %d sm_low %d sm_high %d grid.x %d grid.y %d grid.z %d blocks %d\n", a, b, num, sm_low, sm_high, grid.x, grid.y, grid.z, blocks);
//            printf("worker block %d chose %d sm abandoned\n", blockIdx.x, get_smid());
        }
//        else {
//            printf("worker block %d chose %d sm saved\n", blockIdx.x, get_smid());
//        }
    }
    __syncthreads();
    if(terminate)
        return;
    // do jobs iterately
    __shared__ int index;
    index = 0;
    while(index < blocks)
    {
        // detect if finished blocks over boundary
        if(leader)
        {
            index = atomicAdd(&finished, ITERS);
//            printf("block %d claim real block %d\n", blockIdx.x, index);
            if(index >= blocks) {
                terminate = true;
            }
        }
        __syncthreads();
        if(terminate)
            return;
        int high_boundary = min(index + ITERS, blocks);
        for(int i = index; i < high_boundary; i++)
        {
//            if(leader)
//            {
//                printf("worker block %d start do real block %d\n", blockIdx.x, i);
//            }
            uint3 blockIDX = make_uint3( i % grid.x, (i / grid.x) % grid.y, (i / (grid.x * grid.y)));
            vecAdd(a, b, num, blockIDX, grid);
        }
    }
}

__device__ void vecAdd(int *a, int *b, int num, uint3 blockIDX, dim3 gridDIM) {
    int skip = gridDIM.x * blockDim.x;
    for(int i= threadIdx.x + blockIDX.x * blockDim.x; i< num; i+= skip){
        b[i] = a[i] + b[i];
    }
}

extern "C" __global__ void vecAdd(int *a, int *b, int num) {
    int skip = gridDim.x + blockDim.x;
    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < num; i+= skip) {
        b[i] = a[i] + b[i];
    }
}