/*=====================original==================*/
#define BLOCK_SIZE 16
extern "C" __global__ void MatrixMulCUDA(float *C, float *A,
                                         float *B, int wA,
                                         int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}


/*====================mgpu====================*/
#define SMID_MASK 0xf
#define ITERS_MASK 0xf0
#define WORKER_MASK 0xff00
#define ITERS 4
#define MAX_SM 6
#define GET_SID(flags) (flags & SMID_MASK)
#define GET_WID(flags) (( flags & WORKER_MASK) >> 8)
#define SET_WID(flags, worker) (flags = (flags & ~(WORKER_MASK)) + ((worker) << 8))
#define GET_ITERS(flags) ((flags & ITERS_MASK) >> 4)
#define SET_ITERS(flags, times) (flags = (flags & ~(ITERS_MASK)) + ((times) << 4))
#define IS_LEAD_THREAD (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)

// __device__ int sms_flag;// block limits per sm - max sm id - min sm id
// __device__ int b_cnts; // total block counts
// __device__ int b_fins; // finished block counts
// __device__ dim3 gridDIM; // origin grid dim
// __device__ int w_cnts[MAX_SM]; // workers per sm
__device__ int configs[6 + MAX_SM]; // 0: sms_flag; 1: b_cnts; 2: b_fins; 3~5 : gridDIM; 6~end w_cnts[MAX_SM]
#define SMS_FLAG (configs[0])
#define TOTAL_CNTS (configs[1])
#define FIN_CNTS (configs[2])
#define GridDim_X (configs[3])
#define GridDim_Y (configs[4])
#define GridDim_Z (configs[5])
#define WORKER_ADDR (configs + 6)
#define GET_SM_LOW(sms_flag) ((sms_flag) & 0xff)
#define GET_SM_HIGH(sms_flag) (((sms_flag) & 0xff00) >> 8)
#define GET_BLOCK_LIMIT(sms_flag) (((sms_flag) & 0xffff0000) >> 16)

__device__ void MatrixMulCUDA_V(float *C, float *A, float *B, int wA, int wB, uint3 blockIDX);

extern "C" __global__ void MatrixMulCUDA_V1(float *C, float *A,
                                            float *B, int wA,
                                            int wB) {
    __shared__ int start_block;
    __shared__ int flags;
    // set sid & wid
    if(IS_LEAD_THREAD) {
        int worker;
        start_block = 0;
        flags = 0;
        asm("mov.u32 %0, %smid;":"=r"(flags));
        // sm check
        if(GET_SID(flags) < GET_SM_LOW(SMS_FLAG) || GET_SID(flags) > GET_SM_HIGH(SMS_FLAG)) {
            start_block = -1;
        } else if((worker = atomicAdd(WORKER_ADDR + GET_SID(flags), 1)) >= GET_BLOCK_LIMIT(SMS_FLAG)) {
            start_block = -1;
        } else {
            SET_WID(flags, worker);
        }
    }
    __syncthreads();
    if(start_block == -1)
        goto end;
    __shared__ uint3 blockIDX;

    while(start_block != -1) {
        if(IS_LEAD_THREAD) {
            // sm check
            if(GET_SID(flags) < GET_SM_LOW(SMS_FLAG) || GET_SID(SMS_FLAG) > GET_SM_HIGH(SMS_FLAG)) {
                start_block = -1;
            } else if(GET_WID(flags) >= GET_BLOCK_LIMIT(SMS_FLAG)) {
                // worker check
                start_block = -1;
            } else {
                start_block = atomicAdd(&FIN_CNTS, ITERS);
                if(start_block >= TOTAL_CNTS)
                    start_block = -1;
                else {
                    SET_ITERS(flags, min(start_block + ITERS, TOTAL_CNTS) - start_block);
                    blockIDX = make_uint3(start_block % GridDim_X, (start_block / GridDim_X) % GridDim_Y, (start_block / (GridDim_X * GridDim_Y)));
                }
            }
        }// if threadIdx.x == 0
        __syncthreads();
        if(start_block == -1) {
            goto end;
        }
#pragma unroll
        for(int i = start_block; i < start_block + GET_ITERS(flags);) {
            MatrixMulCUDA_V(C, A, B, wA, wB, blockIDX);
            i++;
            if(ITERS > 1 && (IS_LEAD_THREAD)) {
                blockIDX = make_uint3(i % GridDim_X, (i / GridDim_X) % GridDim_Y, (i / (GridDim_X * GridDim_Y)));
            }
            __syncthreads();
        }
    }// while
    end:
    if(IS_LEAD_THREAD)
        atomicSub(WORKER_ADDR + GET_SID(flags), 1);
}

__device__ void MatrixMulCUDA_V(float *C, float *A, float *B, int wA, int wB, uint3 blockIDX) {
    // Block index
    int bx = blockIDX.x;
    int by = blockIDX.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}