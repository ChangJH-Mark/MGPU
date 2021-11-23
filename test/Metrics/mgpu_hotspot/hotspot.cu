//
// Created by root on 2021/5/10.
//
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

__device__ void calculate_temp_V(int iteration,  //number of iteration
                                 float *power,   //power input
                                 float *temp_src,    //temperature input/output
                                 float *temp_dst,    //temperature input/output
                                 int grid_cols,  //Col of grid
                                 int grid_rows,  //Row of grid
                                 int border_cols,  // border offset
                                 int border_rows,  // border offset
                                 float Cap,      //Capacitance
                                 float Rx,
                                 float Ry,
                                 float Rz,
                                 float step,
                                 float time_elapsed,
                                 uint3 blockIDX);

extern "C" __global__ void calculate_temp_V1(int iteration,  //number of iteration
                                                          float *power,   //power input
                                                          float *temp_src,    //temperature input/output
                                                          float *temp_dst,    //temperature input/output
                                                          int grid_cols,  //Col of grid
                                                          int grid_rows,  //Row of grid
                                                          int border_cols,  // border offset
                                                          int border_rows,  // border offset
                                                          float Cap,      //Capacitance
                                                          float Rx,
                                                          float Ry,
                                                          float Rz,
                                                          float step,
                                                          float time_elapsed){
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
            calculate_temp_V(iteration, power, temp_src, temp_dst, grid_cols,
                  grid_rows, border_cols, border_rows, Cap, Rx, Ry,
                  Rz, step, time_elapsed, blockIDX
            );
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
                    calculate_temp_V1<<<start_block, blockDim, 0, tmp>>>(iteration, power, temp_src, temp_dst, grid_cols,
                            grid_rows, border_cols, border_rows, Cap, Rx, Ry,
                            Rz, step, time_elapsed);
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

// original kernel
#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 16
#endif

#define STR_SIZE 256


#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__device__ void calculate_temp_V(int iteration,  //number of iteration
                               float *power,   //power input
                               float *temp_src,    //temperature input/output
                               float *temp_dst,    //temperature input/output
                               int grid_cols,  //Col of grid
                               int grid_rows,  //Row of grid
                               int border_cols,  // border offset
                               int border_rows,  // border offset
                               float Cap,      //Capacitance
                               float Rx,
                               float Ry,
                               float Rz,
                               float step,
                               float time_elapsed,
                               uint3 blockIDX){

    __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE]; // saving temparary temperature result

    float amb_temp = 80.0;
    float step_div_Cap;
    float Rx_1,Ry_1,Rz_1;

    int bx = blockIDX.x;
    int by = blockIDX.y;

    int tx=threadIdx.x;
    int ty=threadIdx.y;

    step_div_Cap=step/Cap;

    Rx_1=1/Rx;
    Ry_1=1/Ry;
    Rz_1=1/Rz;

    // each block finally computes result for a small block
    // after N iterations.
    // it is the non-overlapping small blocks that cover
    // all the input data

    // calculate the small block size
    int small_block_rows = BLOCK_SIZE-iteration*2;//EXPAND_RATE
    int small_block_cols = BLOCK_SIZE-iteration*2;//EXPAND_RATE

    // calculate the boundary for the block according to
    // the boundary of its small block
    int blkY = small_block_rows*by-border_rows;
    int blkX = small_block_cols*bx-border_cols;
    int blkYmax = blkY+BLOCK_SIZE-1;
    int blkXmax = blkX+BLOCK_SIZE-1;

    // calculate the global thread coordination
    int yidx = blkY+ty;
    int xidx = blkX+tx;

    // load data if it is within the valid input range
    int loadYidx=yidx, loadXidx=xidx;
    int index = grid_cols*loadYidx+loadXidx;

    if(IN_RANGE(loadYidx, 0, grid_rows-1) && IN_RANGE(loadXidx, 0, grid_cols-1)){
        temp_on_cuda[ty][tx] = temp_src[index];  // Load the temperature data from global memory to shared memory
        power_on_cuda[ty][tx] = power[index];// Load the power data from global memory to shared memory
    }
    __syncthreads();

    // effective range within this block that falls within
    // the valid range of the input data
    // used to rule out computation outside the boundary.
    int validYmin = (blkY < 0) ? -blkY : 0;
    int validYmax = (blkYmax > grid_rows-1) ? BLOCK_SIZE-1-(blkYmax-grid_rows+1) : BLOCK_SIZE-1;
    int validXmin = (blkX < 0) ? -blkX : 0;
    int validXmax = (blkXmax > grid_cols-1) ? BLOCK_SIZE-1-(blkXmax-grid_cols+1) : BLOCK_SIZE-1;

    int N = ty-1;
    int S = ty+1;
    int W = tx-1;
    int E = tx+1;

    N = (N < validYmin) ? validYmin : N;
    S = (S > validYmax) ? validYmax : S;
    W = (W < validXmin) ? validXmin : W;
    E = (E > validXmax) ? validXmax : E;

    bool computed;
    for (int i=0; i<iteration ; i++){
        computed = false;
        if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(ty, i+1, BLOCK_SIZE-i-2) &&  \
                  IN_RANGE(tx, validXmin, validXmax) && \
                  IN_RANGE(ty, validYmin, validYmax) ) {
            computed = true;
            temp_t[ty][tx] =   temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
                                                                      (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0*temp_on_cuda[ty][tx]) * Ry_1 +
                                                                      (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0*temp_on_cuda[ty][tx]) * Rx_1 +
                                                                      (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);

        }
        __syncthreads();
        if(i==iteration-1)
            break;
        if(computed)	 //Assign the computation range
            temp_on_cuda[ty][tx]= temp_t[ty][tx];
        __syncthreads();
    }

    // update the global memory
    // after the last iteration, only threads coordinated within the
    // small block perform the calculation and switch on ``computed''
    if (computed){
        temp_dst[index]= temp_t[ty][tx];
    }
}