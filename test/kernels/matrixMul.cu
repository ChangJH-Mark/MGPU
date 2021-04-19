//
// Created by root on 2021/4/13.
//
#define BLOCK_SIZE 16
#define ITERS 10

__device__ void matrixMul(float *C, float *A,
                          float *B, int wA,
                          int wB, uint3 blockIDX, dim3 gridDIM);

__device__ uint get_smid() {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ int finished = 0;

extern "C" __global__ void matrixMulProxy(float *C, float *A, float *B, int wA, int wB,
                                          int sm_low, int sm_high, dim3 gridDIM, int blocks) {
    // reside on sm (sm >= sm_low && sm < sm_high)
    bool leader = false;
    __shared__ bool terminate;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        leader = true;
    }
    if (leader) {
        terminate = false;
        int sm_id = get_smid();
        if (sm_id < sm_low || sm_id >= sm_high) {
            terminate = true;
        } else {
            printf("worker block %d chose %d sm saved\n", blockIdx.x, get_smid());
        }
    }
    __syncthreads();
    if (terminate)
        return;
    // do jobs iterately
    __shared__ int index;
    index = 0;
    while (index < blocks) {
        // detect if finished blocks over boundary
        if (leader) {
            index = atomicAdd(&finished, ITERS);
            printf("block %d claim real block %d\n", blockIdx.x, index);
            if (index >= blocks) {
                terminate = true;
            }
        }
        __syncthreads();
        if (terminate)
            return;
        int high_boundary = min(index + ITERS, blocks);
        for (int i = index; i < high_boundary; i++) {
            uint3 blockIDX = make_uint3(i % gridDIM.x, (i / gridDIM.x) % gridDIM.y, (i / (gridDIM.x * gridDIM.y)));
            if (leader) {
                printf("worker block %d start do real block x %d y %d z %d\n", blockIdx.x, blockIDX.x, blockIDX.y, blockIDX.z);
            }
            matrixMul(C, A, B, wA, wB, blockIDX, gridDIM);
            __syncthreads();
        }
    }
}

__device__ void matrixMul(float *C, float *A,
                          float *B, int wA,
                          int wB, uint3 blockIDX, dim3 gridDIM) {
    // block index
    int bx = blockIDX.x;
    int by = blockIDX.y;

    // thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA - 1;

    // step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to store the sub-matrix of A
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // load the matrics from device memory to shared memory, each thread load one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
        // make sure the matrics are loaded
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }
        // make sure this iteration finish
        __syncthreads();
    }
    // write back to device memory
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}