//
// Created by root on 2021/6/3.
//
#define ITERS 10
__device__ uint get_smid() {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ int finished = 0;

__device__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t, uint3 blockIDX, dim3 gridDIM);
__device__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t, uint3 blockIDX, dim3 gridDIM);


extern "C" __global__ void Fan1Proxy(float *m_cuda, float *a_cuda, int Size, int t,
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
        }
    }
    __syncthreads();
    if (terminate)
        return;
    __shared__ int index;
    index = 0;
    while (index < blocks) {
        if (leader) {
            index = atomicAdd(&finished, ITERS);
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
            Fan1(m_cuda, a_cuda, Size, t, blockIDX, gridDIM);
            __syncthreads();
        }
    }
}


extern "C" __global__ void Fan2Proxy(float *m_cuda, float *a_cuda, float *b_cuda, int Size, int j1, int t,
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
            Fan2(m_cuda, a_cuda, b_cuda, Size, j1, t, blockIDX, gridDIM);
            __syncthreads();
        }
    }
}

__device__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t, uint3 blockIDX, dim3 gridDIM)
{
    //if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) printf(".");
    //printf("blockIDx.x:%d,threadIdx.x:%d,Size:%d,t:%d,Size-1-t:%d\n",blockIdx.x,threadIdx.x,Size,t,Size-1-t);

    if(threadIdx.x + blockIDX.x * blockDim.x >= Size-1-t) return;
    *(m_cuda+Size*(blockDim.x*blockIDX.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*blockIDX.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
}

__device__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t, uint3 blockIDX, dim3 gridDIM)
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