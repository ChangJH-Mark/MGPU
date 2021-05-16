//
// Created by root on 2021/5/16.
//
// modified code
#define ITERS 10

__device__ uint get_smid() {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

__device__ int finished = 0;

__device__ void
find_index_kernel(double *arrayX, double *arrayY, double *CDF, double *u, double *xj, double *yj, double *weights,
                  int Nparticles, uint3 blockIDX, dim3 gridDIM);

__device__ void
normalize_weights_kernel(double *weights, int Nparticles, double *partial_sums, double *CDF, double *u, int *seed,
                         uint3 blockIDX, dim3 gridDIM);

__device__ void sum_kernel(double *partial_sums, int Nparticles, uint3 blockIDX, dim3 gridDIM);

__device__ void
likelihood_kernel(double *arrayX, double *arrayY, double *xj, double *yj, double *CDF, int *ind, int *objxy,
                  double *likelihood, unsigned char *I, double *u, double *weights, int Nparticles, int countOnes,
                  int max_size, int k, int IszY, int Nfr, int *seed, double *partial_sums, uint3 blockIDX,
                  dim3 gridDIM);


extern "C" __global__ void
find_index_kernelProxy(double *arrayX, double *arrayY, double *CDF, double *u, double *xj, double *yj, double *weights,
                       int Nparticles,
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
                printf("worker block %d start do real block x %d y %d z %d\n", blockIdx.x, blockIDX.x, blockIDX.y,
                       blockIDX.z);
            }
// real kernel
//matrixMul(C, A, B, wA, wB, blockIDX, gridDIM);
            find_index_kernel(arrayX, arrayY, CDF, u, xj, yj, weights, Nparticles, blockIDX, gridDIM);
            __syncthreads();
        }
    }
}


extern "C" __global__ void
normalize_weights_kernelProxy(double *weights, int Nparticles, double *partial_sums, double *CDF, double *u, int *seed,
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
                printf("worker block %d start do real block x %d y %d z %d\n", blockIdx.x, blockIDX.x, blockIDX.y,
                       blockIDX.z);
            }
            normalize_weights_kernel(weights, Nparticles, partial_sums, CDF, u, seed, blockIDX, gridDIM);
            __syncthreads();
        }
    }
}


extern "C" __global__ void sum_kernelProxy(double *partial_sums, int Nparticles,
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
                printf("worker block %d start do real block x %d y %d z %d\n", blockIdx.x, blockIDX.x, blockIDX.y,
                       blockIDX.z);
            }
// real kernel
            sum_kernel(partial_sums, Nparticles, blockIDX, gridDIM);
            __syncthreads();
        }
    }
}


extern "C" __global__ void
likelihood_kernelProxy(double *arrayX, double *arrayY, double *xj, double *yj, double *CDF, int *ind, int *objxy,
                       double *likelihood, unsigned char *I, double *u, double *weights, int Nparticles, int countOnes,
                       int max_size, int k, int IszY, int Nfr, int *seed, double *partial_sums,
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
                printf("worker block %d start do real block x %d y %d z %d\n", blockIdx.x, blockIDX.x, blockIDX.y,
                       blockIDX.z);
            }
// real kernel
//matrixMul(C, A, B, wA, wB, blockIDX, gridDIM);
            likelihood_kernel(arrayX, arrayY, xj, yj, CDF, ind, objxy, likelihood, I, u, weights, Nparticles, countOnes,
                              max_size, k, IszY, Nfr, seed, partial_sums, blockIDX, gridDIM);
            __syncthreads();
        }
    }
}

// origin cuda code
const int threads_per_block = 512;
/********************************
 * CALC LIKELIHOOD SUM
 * DETERMINES THE LIKELIHOOD SUM BASED ON THE FORMULA: SUM( (IK[IND] - 100)^2 - (IK[IND] - 228)^2)/ 100
 * param 1 I 3D matrix
 * param 2 current ind array
 * param 3 length of ind array
 * returns a double representing the sum
 ********************************/
__device__ double calcLikelihoodSum(unsigned char *I, int *ind, int numOnes, int index) {
    double likelihoodSum = 0.0;
    int x;
    for (x = 0; x < numOnes; x++)
        likelihoodSum += (pow((double) (I[ind[index * numOnes + x]] - 100), 2) -
                          pow((double) (I[ind[index * numOnes + x]] - 228), 2)) / 50.0;
    return likelihoodSum;
}

/****************************
CDF CALCULATE
CALCULATES CDF
param1 CDF
param2 weights
param3 Nparticles
 *****************************/
__device__ void cdfCalc(double *CDF, double *weights, int Nparticles) {
    int x;
    CDF[0] = weights[0];
    for (x = 1; x < Nparticles; x++) {
        CDF[x] = weights[x] + CDF[x - 1];
    }
}

/*****************************
 * RANDU
 * GENERATES A UNIFORM DISTRIBUTION
 * returns a double representing a randomily generated number from a uniform distribution with range [0, 1)
 ******************************/
__device__ double d_randu(int *seed, int index) {

    int M = INT_MAX;
    int A = 1103515245;
    int C = 12345;
    int num = A * seed[index] + C;
    seed[index] = num % M;

    return fabs(seed[index] / ((double) M));
}

__device__ double d_randn(int *seed, int index) {
    //Box-Muller algortihm
    double pi = 3.14159265358979323846;
    double u = d_randu(seed, index);
    double v = d_randu(seed, index);
    double cosine = cos(2 * pi * v);
    double rt = -2 * log(u);
    return sqrt(rt) * cosine;
}


/****************************
UPDATE WEIGHTS
UPDATES WEIGHTS
param1 weights
param2 likelihood
param3 Nparcitles
 ****************************/
__device__ double updateWeights(double *weights, double *likelihood, int Nparticles) {
    int x;
    double sum = 0;
    for (x = 0; x < Nparticles; x++) {
        weights[x] = weights[x] * exp(likelihood[x]);
        sum += weights[x];
    }
    return sum;
}

__device__ int findIndexBin(double *CDF, int beginIndex, int endIndex, double value) {
    if (endIndex < beginIndex)
        return -1;
    int middleIndex;
    while (endIndex > beginIndex) {
        middleIndex = beginIndex + ((endIndex - beginIndex) / 2);
        if (CDF[middleIndex] >= value) {
            if (middleIndex == 0)
                return middleIndex;
            else if (CDF[middleIndex - 1] < value)
                return middleIndex;
            else if (CDF[middleIndex - 1] == value) {
                while (CDF[middleIndex] == value && middleIndex >= 0)
                    middleIndex--;
                middleIndex++;
                return middleIndex;
            }
        }
        if (CDF[middleIndex] > value)
            endIndex = middleIndex - 1;
        else
            beginIndex = middleIndex + 1;
    }
    return -1;
}

/** added this function. was missing in original double version.
 * Takes in a double and returns an integer that approximates to that double
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */
__device__ double dev_round_double(double value) {
    int newValue = (int) (value);
    if (value - newValue < .5f)
        return newValue;
    else
        return newValue++;
}


/*****************************
 * CUDA Find Index Kernel Function to replace FindIndex
 * param1: arrayX
 * param2: arrayY
 * param3: CDF
 * param4: u
 * param5: xj
 * param6: yj
 * param7: weights
 * param8: Nparticles
 *****************************/
__device__ void
find_index_kernel(double *arrayX, double *arrayY, double *CDF, double *u, double *xj, double *yj, double *weights,
                  int Nparticles, uint3 blockIDX, dim3 gridDIM) {
    int block_id = blockIDX.x;
    int i = blockDim.x * block_id + threadIdx.x;

    if (i < Nparticles) {

        int index = -1;
        int x;

        for (x = 0; x < Nparticles; x++) {
            if (CDF[x] >= u[i]) {
                index = x;
                break;
            }
        }
        if (index == -1) {
            index = Nparticles - 1;
        }

        xj[i] = arrayX[index];
        yj[i] = arrayY[index];

        //weights[i] = 1 / ((double) (Nparticles)); //moved this code to the beginning of likelihood kernel

    }
    __syncthreads();
}

__device__ void
normalize_weights_kernel(double *weights, int Nparticles, double *partial_sums, double *CDF, double *u, int *seed,
                         uint3 blockIDX, dim3 gridDIM) {
    int block_id = blockIDX.x;
    int i = blockDim.x * block_id + threadIdx.x;
    __shared__ double u1, sumWeights;

    if (0 == threadIdx.x)
        sumWeights = partial_sums[0];

    __syncthreads();

    if (i < Nparticles) {
        weights[i] = weights[i] / sumWeights;
    }

    __syncthreads();

    if (i == 0) {
        cdfCalc(CDF, weights, Nparticles);
        u[0] = (1 / ((double) (Nparticles))) *
               d_randu(seed, i); // do this to allow all threads in all blocks to use the same u1
    }

    __syncthreads();

    if (0 == threadIdx.x)
        u1 = u[0];

    __syncthreads();

    if (i < Nparticles) {
        u[i] = u1 + i / ((double) (Nparticles));
    }
}

__device__ void sum_kernel(double *partial_sums, int Nparticles, uint3 blockIDX, dim3 gridDIM) {
    int block_id = blockIDX.x;
    int i = blockDim.x * block_id + threadIdx.x;

    if (i == 0) {
        int x;
        double sum = 0.0;
        int num_blocks = ceil((double) Nparticles / (double) threads_per_block);
        for (x = 0; x < num_blocks; x++) {
            sum += partial_sums[x];
        }
        partial_sums[0] = sum;
    }
}

/*****************************
 * CUDA Likelihood Kernel Function to replace FindIndex
 * param1: arrayX
 * param2: arrayY
 * param2.5: CDF
 * param3: ind
 * param4: objxy
 * param5: likelihood
 * param6: I
 * param6.5: u
 * param6.75: weights
 * param7: Nparticles
 * param8: countOnes
 * param9: max_size
 * param10: k
 * param11: IszY
 * param12: Nfr
 *****************************/
__device__ void
likelihood_kernel(double *arrayX, double *arrayY, double *xj, double *yj, double *CDF, int *ind, int *objxy,
                  double *likelihood, unsigned char *I, double *u, double *weights, int Nparticles, int countOnes,
                  int max_size, int k, int IszY, int Nfr, int *seed, double *partial_sums, uint3 blockIDX,
                  dim3 gridDIM) {
    int block_id = blockIDX.x;
    int i = blockDim.x * block_id + threadIdx.x;
    int y;

    int indX, indY;
    __shared__ double buffer[512];
    if (i < Nparticles) {
        arrayX[i] = xj[i];
        arrayY[i] = yj[i];

        weights[i] = 1 /
                     ((double) (Nparticles)); //Donnie - moved this line from end of find_index_kernel to prevent all weights from being reset before calculating position on final iteration.

        arrayX[i] = arrayX[i] + 1.0 + 5.0 * d_randn(seed, i);
        arrayY[i] = arrayY[i] - 2.0 + 2.0 * d_randn(seed, i);

    }

    __syncthreads();

    if (i < Nparticles) {
        for (y = 0; y < countOnes; y++) {
            //added dev_round_double() to be consistent with roundDouble
            indX = dev_round_double(arrayX[i]) + objxy[y * 2 + 1];
            indY = dev_round_double(arrayY[i]) + objxy[y * 2];

            ind[i * countOnes + y] = abs(indX * IszY * Nfr + indY * Nfr + k);
            if (ind[i * countOnes + y] >= max_size)
                ind[i * countOnes + y] = 0;
        }
        likelihood[i] = calcLikelihoodSum(I, ind, countOnes, i);

        likelihood[i] = likelihood[i] / countOnes;

        weights[i] = weights[i] * exp(likelihood[i]); //Donnie Newell - added the missing exponential function call

    }

    buffer[threadIdx.x] = 0.0;

    __syncthreads();

    if (i < Nparticles) {

        buffer[threadIdx.x] = weights[i];
    }

    __syncthreads();

    //this doesn't account for the last block that isn't full
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            buffer[threadIdx.x] += buffer[threadIdx.x + s];
        }

        __syncthreads();

    }
    if (threadIdx.x == 0) {
        partial_sums[blockIDX.x] = buffer[0];
    }

    __syncthreads();


}