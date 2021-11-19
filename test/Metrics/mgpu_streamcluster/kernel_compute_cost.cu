//
// Created by root on 2021/5/7.
//
#include "streamcluster_header.cu"

/*====================mgpu====================*/
#define SMID_MASK 0xff
#define ITERS_MASK 0xff00
#define WORKER_MASK 0xff0000
#define ITERS 5
#define MAX_SM 6
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

__device__ void
kernel_compute_cost_V(int num, int dim, long x, Point *p, int K, int stride,
                    float *coord_d, float *work_mem_d, int *center_table_d, bool *switch_membership_d, uint3 blockIDX);


extern "C" __global__ void kernel_compute_cost_V1(int num, int dim, long x, Point *p, int K, int stride,
                                                    float *coord_d, float *work_mem_d, int *center_table_d,
                                                    bool *switch_membership_d) {
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
            kernel_compute_cost_V(num, dim, x, p, K, stride, coord_d, work_mem_d, center_table_d, switch_membership_d, blockIDX);
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
                    kernel_compute_cost_V1<<<start_block, blockDim, 0, tmp>>>(num, dim, x, p, K, stride, coord_d, work_mem_d, center_table_d, switch_membership_d);
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

//=======================================
// Euclidean Distance
//=======================================
__device__ float
d_dist(int p1, int p2, int num, int dim, float *coord_d) {
    float retval = 0.0;
    for (int i = 0; i < dim; i++) {
        float tmp = coord_d[(i * num) + p1] - coord_d[(i * num) + p2];
        retval += tmp * tmp;
    }
    return retval;
}

//=======================================
// Kernel - Compute Cost
//=======================================
__device__ void
kernel_compute_cost_V(int num, int dim, long x, Point *p, int K, int stride,
                    float *coord_d, float *work_mem_d, int *center_table_d, bool *switch_membership_d, uint3 blockIDX
                    ) {
    // block ID and global thread ID
    const int bid = blockIDX.x + GridDim_X * blockIDX.y;
    const int tid = blockDim.x * bid + threadIdx.x;

    if (tid < num) {
        float *lower = &work_mem_d[tid * stride];

        // cost between this point and point[x]: euclidean distance multiplied by weight
        float x_cost = d_dist(tid, x, num, dim, coord_d) * p[tid].weight;

        // if computed cost is less then original (it saves), mark it as to reassign
        if (x_cost < p[tid].cost) {
            switch_membership_d[tid] = 1;
            lower[K] += x_cost - p[tid].cost;
        }
            // if computed cost is larger, save the difference
        else {
            lower[center_table_d[p[tid].assign]] += p[tid].cost - x_cost;
        }
    }
}
