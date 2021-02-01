/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

 #include <stdio.h>

 // For the CUDA runtime routines (prefixed with "cuda_")
 #include <cuda_runtime.h>
 #include "allocator.h"
 /**
  * CUDA Kernel Device code
  *
  * Computes the vector addition of A and B into C. The 3 vectors have the same
  * number of elements numElements.
  */
 __global__ void
 vectorAdd(const float *A, const float *B, float *C, int numElements)
 {
     int i = blockDim.x * blockIdx.x + threadIdx.x;
 
     if (i < numElements)
     {
         C[i] = A[i] + B[i];
     }
 }
 
 /**
  * Host main routine
  */
 int
 main(void)
 {
     // Print the vector length to be used, and compute its size
     int numElements = 50000;
     size_t size = numElements * sizeof(float);
     printf("[Vector addition of %d elements]\n", numElements);
 
     // Allocate the host input vector A
    //  float *h_A = (float *)malloc(size);
    err_t e;
    auto h_A = MemAlloc(size, CPUNOPINNOMAP, &e);

 
     // Allocate the host input vector B
    //  float *h_B = (float *)malloc(size);
     auto h_B = MemAlloc(size, CPUNOPINNOMAP, &e);
 
     // Allocate the host output vector C
    //  float *h_C = (float *)malloc(size);
     auto h_C = MemAlloc(size, CPUNOPINNOMAP, &e);
 
     // Verify that allocations succeeded
     if(e != UM_SUCCESS) {
         fprintf(stderr, "fail to allocate memory");
         exit(EXIT_FAILURE);
     }
 
     // Initialize the host input vectors
     for (int i = 0; i < numElements; ++i)
     {
         h_A.hostAddr()[i] = rand()/(float)RAND_MAX;
         h_B.hostAddr()[i] = rand()/(float)RAND_MAX;
     }
 
     // Allocate the device input vector A
    auto d_A = MemAlloc(size, GPUMEM, &e);
 
     if (e != cudaSuccess)
     {
         fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(cudaError_t(e)));
         exit(EXIT_FAILURE);
     }
 
     // Allocate the device input vector B
     auto d_B = MemAlloc(size, GPUMEM, &e);
 
     if (e != cudaSuccess)
     {
         fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(cudaError_t(e)));
         exit(EXIT_FAILURE);
     }
 
     // Allocate the device output vector C
     auto d_C = MemAlloc(size, GPUMEM, &e);
 
     if (e != cudaSuccess)
     {
         fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(cudaError_t(e)));
         exit(EXIT_FAILURE);
     }
 
     // Copy the host input vectors A and B in host memory to the device input vectors in
     // device memory
     printf("Copy input data from the host memory to the CUDA device\n");
     auto err = cudaMemcpy(d_A.deviceAddr(), h_A.hostAddr(), size, cudaMemcpyHostToDevice);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     err = cudaMemcpy(d_B.deviceAddr(), h_B.hostAddr(), size, cudaMemcpyHostToDevice);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     // Launch the Vector Add CUDA Kernel
     int threadsPerBlock = 256;
     int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
     printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
     vectorAdd<<<blocksPerGrid, threadsPerBlock>>>((float*)d_A.deviceAddr(), (float*)d_B.deviceAddr(), (float *)d_C.deviceAddr(), numElements);
     err = cudaGetLastError();
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(cudaError_t(err)));
         exit(EXIT_FAILURE);
     }
 
     // Copy the device result vector in device memory to the host result vector
     // in host memory.
     printf("Copy output data from the CUDA device to the host memory\n");
     err = cudaMemcpy(h_C.hostAddr(), d_C.deviceAddr(), size, cudaMemcpyDeviceToHost);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(cudaError_t(err)));
         exit(EXIT_FAILURE);
     }
 
     // Verify that the result vector is correct
     for (int i = 0; i < numElements; ++i)
     {
         if (fabs(h_A.hostAddr()[i] + h_B.hostAddr()[i] - h_C.hostAddr()[i]) > 1e-5)
         {
             fprintf(stderr, "Result verification failed at element %d!\n", i);
             exit(EXIT_FAILURE);
         }
     }
 
     printf("Test PASSED\n");
 
     // Free device global memory
     err =(cudaError_t)  d_A.free();
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(cudaError_t(err)));
         exit(EXIT_FAILURE);
     }
 
     err = (cudaError_t)d_B.free();
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(cudaError_t(err)));
         exit(EXIT_FAILURE);
     }
 
     err = (cudaError_t) d_C.free();
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(cudaError_t(err)));
         exit(EXIT_FAILURE);
     }
 
     // Free host memory
     h_A.free();
     h_B.free();
     h_C.free();
 
     printf("Done\n");
     return 0;
 }
 