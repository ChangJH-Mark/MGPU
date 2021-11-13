//
// Created by root on 2021/6/3.
//
/*====================mgpu====================*/
#define SMID_MASK 0xff
#define ITERS_MASK 0xff00
#define WORKER_MASK 0xff0000
#define ITERS 10
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

__device__ void Fan1_V(float *m_cuda, float *a_cuda, int Size, int t, uint3 blockIDX);
__device__ void Fan2_V(float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t, uint3 blockIDX);


extern "C" __global__ void Fan1_V1(float *m_cuda, float *a_cuda, int Size, int t) {
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
            Fan1_V(m_cuda, a_cuda, Size, t, blockIDX);
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
                    Fan1_V1<<<start_block, blockDim, 0, tmp>>>(m_cuda, a_cuda, Size, t);
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


extern "C" __global__ void Fan2_V1(float *m_cuda, float *a_cuda, float *b_cuda, int Size, int j1, int t) {
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
            Fan2_V(m_cuda, a_cuda, b_cuda, Size, j1, t, blockIDX);
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
                    Fan2_V1<<<start_block, blockDim, 0, tmp>>>(m_cuda, a_cuda, b_cuda, Size, j1, t);
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

__device__ void Fan1_V(float *m_cuda, float *a_cuda, int Size, int t, uint3 blockIDX)
{
    //if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) printf(".");
    //printf("blockIDx.x:%d,threadIdx.x:%d,Size:%d,t:%d,Size-1-t:%d\n",blockIdx.x,threadIdx.x,Size,t,Size-1-t);

    if(threadIdx.x + blockIDX.x * blockDim.x >= Size-1-t) return;
    *(m_cuda+Size*(blockDim.x*blockIDX.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*blockIDX.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
}

__device__ void Fan2_V(float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t, uint3 blockIDX)
{
    if(threadIdx.x + blockIDX.x * blockDim.x >= Size-1-t) return;
    if(threadIdx.y + blockIDX.y * blockDim.y >= Size-t) return;

    int xidx = blockIDX.x * blockDim.x + threadIdx.x;
    int yidx = blockIDX.y * blockDim.y + threadIdx.y;
    //printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);

    a_cuda[Size*(xidx+1+t)+(yidx+t)] -= m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
    //a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
    if(yidx == 0){
        //printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
        //printf("xidx:%d,yidx:%d\n",xidx,yidx);
        b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
    }
}