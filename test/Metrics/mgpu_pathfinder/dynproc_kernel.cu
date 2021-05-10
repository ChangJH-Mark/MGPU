//
// Created by root on 2021/4/27.
//
#include <stdio.h>
#define BLOCK_SIZE 256
#define HALO 1 // halo width along one direction when advancing to the next iteration
#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))
#define ITERS 10

__device__ void dynproc_kernel(
        int iteration,
        int *gpuWall,
        int *gpuSrc,
        int *gpuResults,
        int cols,
        int rows,
        int startStep,
        int border,
        uint3 blockIDX);
__device__ uint get_smid() {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ int finished = 0;
extern "C" __global__ void dynproc_kernelProxy(
        int iteration,
        int *gpuWall,
        int *gpuSrc,
        int *gpuResults,
        int cols,
        int rows,
        int startStep,
        int border,
        int sm_low,
        int sm_high,
        dim3 grid,
        int blocks
        )
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
            printf("worker block %d chose %d sm abandoned\n", blockIdx.x, get_smid());
        }
        else {
            printf("worker block %d chose %d sm saved\n", blockIdx.x, get_smid());
        }
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
            printf("block %d claim real block %d\n", blockIdx.x, index);
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
            if(leader)
            {
                printf("worker block %d start do real block %d\n", blockIdx.x, i);
            }
            uint3 blockIDX = make_uint3( i % grid.x, (i / grid.x) % grid.y, (i / (grid.x * grid.y)));
            dynproc_kernel(iteration,gpuWall,gpuSrc,gpuResults,cols,rows,startStep,border, blockIDX);
        }
    }
}

__device__ void dynproc_kernel(
        int iteration,
        int *gpuWall,
        int *gpuSrc,
        int *gpuResults,
        int cols,
        int rows,
        int startStep,
        int border,
        uint3 blockIDX)
{

    __shared__ int prev[BLOCK_SIZE];
    __shared__ int result[BLOCK_SIZE];

    int bx = blockIDX.x;
    int tx=threadIdx.x;

    // each block finally computes result for a small block
    // after N iterations.
    // it is the non-overlapping small blocks that cover
    // all the input data

    // calculate the small block size
    int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

    // calculate the boundary for the block according to
    // the boundary of its small block
    int blkX = small_block_cols*bx-border;
    int blkXmax = blkX+BLOCK_SIZE-1;

    // calculate the global thread coordination
    int xidx = blkX+tx;

    // effective range within this block that falls within
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

    int W = tx-1;
    int E = tx+1;

    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool isValid = IN_RANGE(tx, validXmin, validXmax);

    if(IN_RANGE(xidx, 0, cols-1)){
        prev[tx] = gpuSrc[xidx];
    }
    __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    bool computed;
    for (int i=0; i<iteration ; i++){
        computed = false;
        if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  isValid){
            computed = true;
            int left = prev[W];
            int up = prev[tx];
            int right = prev[E];
            int shortest = MIN(left, up);
            shortest = MIN(shortest, right);
            int index = cols*(startStep+i)+xidx;
            result[tx] = shortest + gpuWall[index];

        }
        __syncthreads();
        if(i==iteration-1)
            break;
        if(computed)	 //Assign the computation range
            prev[tx]= result[tx];
        __syncthreads(); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
    }

    // update the global memory
    // after the last iteration, only threads coordinated within the
    // small block perform the calculation and switch on ``computed''
    if (computed){
        gpuResults[xidx]=result[tx];
    }
}