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
    auto h_A = MemAlloc(size, CPUNOPINNOMAP);

 
     // Allocate the host input vector B
    //  float *h_B = (float *)malloc(size);
     auto h_B = MemAlloc(size, CPUNOPINNOMAP);
 
     // Allocate the host output vector C
    //  float *h_C = (float *)malloc(size);
     auto h_C = MemAlloc(size, CPUNOPINNOMAP);
 
     // Verify that allocations succeeded
     if(h_A.err != UM_SUCCESS) {
         fprintf(stderr, "fail to allocate memory");
         exit(EXIT_FAILURE);
     }
 
     // Initialize the host input vectors
     for (int i = 0; i < numElements; ++i)
     {
         h_A.uptr.cpu_address[i] = rand()/(float)RAND_MAX;
         h_B.uptr.cpu_address[i] = rand()/(float)RAND_MAX;
     }
 
     // Allocate the device input vector A
     float *d_A = NULL;
    auto  err = cudaMalloc((void **)&d_A, size);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     // Allocate the device input vector B
     float *d_B = NULL;
     err = cudaMalloc((void **)&d_B, size);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     // Allocate the device output vector C
     float *d_C = NULL;
     err = cudaMalloc((void **)&d_C, size);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     // Copy the host input vectors A and B in host memory to the device input vectors in
     // device memory
     printf("Copy input data from the host memory to the CUDA device\n");
     err = cudaMemcpy(d_A, h_A.uptr.cpu_address, size, cudaMemcpyHostToDevice);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     err = cudaMemcpy(d_B, h_B.uptr.cpu_address, size, cudaMemcpyHostToDevice);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     // Launch the Vector Add CUDA Kernel
     int threadsPerBlock = 256;
     int blocksPerGrid =(numElements + threadsPerBlock - 1) / threadsPerBlock;
     printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
     vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
     err = cudaGetLastError();
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     // Copy the device result vector in device memory to the host result vector
     // in host memory.
     printf("Copy output data from the CUDA device to the host memory\n");
     err = cudaMemcpy(h_C.uptr.cpu_address, d_C, size, cudaMemcpyDeviceToHost);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     // Verify that the result vector is correct
     for (int i = 0; i < numElements; ++i)
     {
         if (fabs(h_A.uptr.cpu_address[i] + h_B.uptr.cpu_address[i] - h_C.uptr.cpu_address[i]) > 1e-5)
         {
             fprintf(stderr, "Result verification failed at element %d!\n", i);
             exit(EXIT_FAILURE);
         }
     }
 
     printf("Test PASSED\n");
 
     // Free device global memory
     err = cudaFree(d_A);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     err = cudaFree(d_B);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     err = cudaFree(d_C);
 
     if (err != cudaSuccess)
     {
         fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     // Free host memory
     h_A.uptr.free();
     h_B.uptr.free();
     h_C.uptr.free();
 
     printf("Done\n");
     return 0;
 }
 