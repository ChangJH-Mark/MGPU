//
// Created by root on 2021/5/7.
//
#include "streamcluster_header.cu"
#define ITERS 10

__device__ uint get_smid() {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ int finished = 0;

__device__ void
kernel_compute_cost(int num, int dim, long x, Point *p, int K, int stride,
                    float *coord_d, float *work_mem_d, int *center_table_d, bool *switch_membership_d, uint3 blockIDX,
                    dim3 gridDIM);


extern "C" __global__ void kernel_compute_costProxy(int num, int dim, long x, Point *p, int K, int stride,
                                                    float *coord_d, float *work_mem_d, int *center_table_d,
                                                    bool *switch_membership_d,
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
//            printf("worker block %d chose %d sm saved\n", blockIdx.x, get_smid());
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
//            printf("block %d claim real block %d\n", blockIdx.x, index);
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
//            if (leader) {
//                printf("worker block %d start do real block x %d y %d z %d\n", blockIdx.x, blockIDX.x, blockIDX.y,
//                       blockIDX.z);
//            }
// real kernel
//matrixMul(C, A, B, wA, wB, blockIDX, gridDIM);
            kernel_compute_cost(num, dim, x, p, K, stride,
                                coord_d, work_mem_d, center_table_d, switch_membership_d, blockIDX, gridDIM);
            __syncthreads();
        }
    }
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
kernel_compute_cost(int num, int dim, long x, Point *p, int K, int stride,
                    float *coord_d, float *work_mem_d, int *center_table_d, bool *switch_membership_d, uint3 blockIDX,
                    dim3 gridDIM) {
    // block ID and global thread ID
    const int bid = blockIDX.x + gridDIM.x * blockIDX.y;
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
