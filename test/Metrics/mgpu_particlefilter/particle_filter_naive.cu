//
// Created by root on 2021/5/16.
//
// modify code
#define ITERS 10
__device__ uint get_smid() {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ int finished = 0;
__device__ void kernel(double * arrayX, double * arrayY, double * CDF, double * u, double * xj, double * yj, int Nparticles, uint3 blockIDX, dim3 gridDIM);

extern "C" __global__ void kernelProxy(double * arrayX, double * arrayY, double * CDF, double * u, double * xj, double * yj, int Nparticles,
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
// real kernel
            kernel(arrayX, arrayY, CDF, u, xj, yj, Nparticles, blockIDX, gridDIM);
            __syncthreads();
        }
    }
}
// original code
__device__ int findIndexSeq(double * CDF, int lengthCDF, double value)
{
    int index = -1;
    int x;
    for(x = 0; x < lengthCDF; x++)
    {
        if(CDF[x] >= value)
        {
            index = x;
            break;
        }
    }
    if(index == -1)
        return lengthCDF-1;
    return index;
}
__device__ int findIndexBin(double * CDF, int beginIndex, int endIndex, double value)
{
    if(endIndex < beginIndex)
        return -1;
    int middleIndex;
    while(endIndex > beginIndex)
    {
        middleIndex = beginIndex + ((endIndex-beginIndex)/2);
        if(CDF[middleIndex] >= value)
        {
            if(middleIndex == 0)
                return middleIndex;
            else if(CDF[middleIndex-1] < value)
                return middleIndex;
            else if(CDF[middleIndex-1] == value)
            {
                while(CDF[middleIndex] == value && middleIndex >= 0)
                    middleIndex--;
                middleIndex++;
                return middleIndex;
            }
        }
        if(CDF[middleIndex] > value)
            endIndex = middleIndex-1;
        else
            beginIndex = middleIndex+1;
    }
    return -1;
}
/*****************************
* CUDA Kernel Function to replace FindIndex
* param1: arrayX
* param2: arrayY
* param3: CDF
* param4: u
* param5: xj
* param6: yj
* param7: Nparticles
*****************************/
__device__ void kernel(double * arrayX, double * arrayY, double * CDF, double * u, double * xj, double * yj, int Nparticles, uint3 blockIDX, dim3 gridDIM){
    int block_id = blockIDX.x;// + gridDim.x * blockIdx.y;
    int i = blockDim.x * block_id + threadIdx.x;

    if(i < Nparticles){

        int index = -1;
        int x;

        for(x = 0; x < Nparticles; x++){
            if(CDF[x] >= u[i]){
                index = x;
                break;
            }
        }
        if(index == -1){
            index = Nparticles-1;
        }

        xj[i] = arrayX[index];
        yj[i] = arrayY[index];

    }
}