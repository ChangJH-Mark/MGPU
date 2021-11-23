//
// Created by root on 2021/4/27.
//

#define BLOCK_SIZE 256
#define HALO 1 // halo width along one direction when advancing to the next iteration
#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__device__ void dynproc_kernel_V(
        int iteration,
        int *gpuWall,
        int *gpuSrc,
        int *gpuResults,
        int cols,
        int rows,
        int startStep,
        int border,
        uint3 blockIDX);

/*====================mgpu====================*/
#define SMID_MASK 0xff
#define ITERS_MASK 0xff00
#define WORKER_MASK 0xff0000
#define ITERS 10
#define GET_SID(flags) (flags & SMID_MASK)
#define GET_WID(flags) (( flags & WORKER_MASK) >> 16)
#define SET_WID(flags, worker) (flags = (flags & ~(WORKER_MASK)) + ((worker) << 16))
#define GET_ITERS(flags) ((flags & ITERS_MASK) >> 8)
#define SET_ITERS(flags, times) (flags = (flags & ~(ITERS_MASK)) + ((times) << 8))
#define IS_LEAD_THREAD (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
#define IS_PRIMARY_WORKER ((flags & WORKER_MASK) == 0)

__device__ int configs[6 + MAX_SM]; // 0: sms_flag; 1: b_cnts; 2: b_fins; 3~5 : gridDIM; 6~end w_cnts[MAX_SM]
#define SMS_FLAG (configs[0])
#define TOTAL_CNTS (__ldca(configs + 1))
#define FIN_CNTS (configs[2])
#define GridDim_X (__ldca(configs + 3))
#define GridDim_Y (__ldca(configs + 4))
#define GridDim_Z (__ldca(configs + 5))
#define WORKER_ADDR (configs + 6)
#define GET_SM_LOW(sms_flag) ((sms_flag) & 0xff)
#define GET_SM_HIGH(sms_flag) (((sms_flag) & 0xff00) >> 8)
#define GET_BLOCK_LIMIT(sms_flag) (((sms_flag) & 0xffff0000) >> 16)

extern "C" __global__ void dynproc_kernel_V1(
        int iteration,
        int *gpuWall,
        int *gpuSrc,
        int *gpuResults,
        int cols,
        int rows,
        int startStep,
        int border) {
    __shared__ int start_block;
    __shared__ int flags;
    __shared__ uint3 blockIDX;
    // set sid & wid
    if (IS_LEAD_THREAD) {
        start_block = 0;
        asm("mov.u32 %0, %smid;":"=r"(flags));
        // tmp use of start_block
        start_block = atomicAdd(WORKER_ADDR + GET_SID(flags), 1);
        SET_WID(flags, start_block);
    }

    __syncthreads();

    while (start_block != -1) {
        if (IS_LEAD_THREAD) {
            start_block = SMS_FLAG;
            // sm check
            if (GET_SID(flags) < GET_SM_LOW(start_block) || GET_SID(flags) > GET_SM_HIGH(start_block)) {
                start_block = -1;
            } else if (GET_WID(flags) >= GET_BLOCK_LIMIT(start_block)) {
                // worker check
                start_block = -1;
            } else {
                start_block = atomicAdd(&FIN_CNTS, ITERS);
                if (start_block >= TOTAL_CNTS)
                    start_block = -1;
                else {
                    SET_ITERS(flags, min(start_block + ITERS, TOTAL_CNTS) - start_block);
                    blockIDX.x = fmodf(start_block , GridDim_X);
                    blockIDX.y = fmodf(__fdividef(start_block, GridDim_X) , GridDim_X);
                    blockIDX.z = __fdividef(start_block, GridDim_X * GridDim_Y);
                }
            }
        }// if threadIdx.x == 0
        __syncthreads();
        if (start_block == -1) {
            goto end;
        }
#pragma unroll
        for (int i = 0; i < GET_ITERS(flags); i++) {
            // calculate blockIDX
            if(i != 0 && IS_LEAD_THREAD) {
                blockIDX.x += 1;
                if(blockIDX.x == GridDim_X)
                {
                    blockIDX.x = 0;
                    blockIDX.y++;
                    if(blockIDX.y == GridDim_Y) {
                        blockIDX.y = 0;
                        blockIDX.z++;
                    }
                }
            }
            __syncthreads();
            dynproc_kernel_V(iteration,gpuWall,gpuSrc,gpuResults,cols,rows,startStep,border, blockIDX);
        }
        if(IS_PRIMARY_WORKER) {
            if(IS_LEAD_THREAD) {
                start_block = FIN_CNTS - (TOTAL_CNTS - ITERS);
                if(start_block >= 0)
                    goto out;

                start_block = GET_BLOCK_LIMIT(SMS_FLAG) - *(WORKER_ADDR + GET_SID(flags));
                if(start_block > 0) {
                    cudaStream_t tmp;
                    cudaStreamCreateWithFlags(&tmp, cudaStreamNonBlocking);
                    dynproc_kernel_V1<<<start_block, blockDim, 0, tmp>>>(iteration,gpuWall,gpuSrc,gpuResults,cols,rows,startStep,border);
                    cudaStreamDestroy(tmp);
                }
                start_block = 0;
            }
            out:
            __syncthreads();
        }
    }// while
    end:
    if (IS_LEAD_THREAD)
        atomicSub(WORKER_ADDR + GET_SID(flags), 1);
}

__device__ void dynproc_kernel_V(
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