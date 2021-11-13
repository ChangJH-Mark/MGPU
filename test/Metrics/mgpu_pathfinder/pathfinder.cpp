//
// Created by root on 2021/4/27.
//
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "client/api.h"

#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

#define BENCH_PRINT

void run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;

//#define BENCH_PRINT


void
init(int argc, char** argv)
{
    if(argc==4){

        cols = atoi(argv[1]);

        rows = atoi(argv[2]);

        pyramid_height=atoi(argv[3]);
    }else{
        printf("Usage: dynproc row_len col_len pyramid_height\n");
        exit(0);
    }
//    ::data = new int[rows*cols];
    ::data = static_cast<int *>(mgpu::cudaMallocHost(sizeof(int) * rows * cols));

    wall = new int*[rows];

    for(int n=0; n<rows; n++)

        wall[n]=::data+cols*n;

//    result = new int[cols];
    result = static_cast<int *>(mgpu::cudaMallocHost(sizeof(int) * cols));



    int seed = M_SEED;

    srand(seed);



    for (int i = 0; i < rows; i++)

    {

        for (int j = 0; j < cols; j++)

        {

            wall[i][j] = rand() % 10;

        }

    }

#ifdef BENCH_PRINT

    for (int i = 0; i < rows; i++)

    {

        for (int j = 0; j < cols; j++)

        {

            printf("%d ",wall[i][j]) ;

        }

        printf("\n") ;

    }

#endif
}

void
fatal(char *s)
{
    fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

/*
   compute N time steps
*/
int calc_path(int *gpuWall, int *gpuResult[2], int rows, int cols, \
	 int pyramid_height, int blockCols, int borderCols)
{
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(blockCols);

    int src = 1, dst = 0;
    for (int t = 0; t < rows-1; t+=pyramid_height) {
        int temp = src;
        src = dst;
        dst = temp;

        mgpu::LaunchConf conf;
        conf.grid = dimGrid; conf.block = dimBlock;
        mgpu::cudaLaunchKernel(conf, "/opt/custom/ptx/pathfinder.ptx","dynproc_kernel", MIN(pyramid_height, rows-t-1),
                               gpuWall, gpuResult[src], gpuResult[dst],
                               cols,rows, t, borderCols);
    }
    return dst;
}

int main(int argc, char** argv)
{
    auto start = std::chrono::steady_clock::now();
    int num_devices;
    num_devices = mgpu::cudaGetDeviceCount();
    if (num_devices > 1) mgpu::cudaSetDevice(DEVICE);

    run(argc,argv);
    auto end = std::chrono::steady_clock::now();
    printf( "cost usec %ld \n", std::chrono::duration_cast<chrono::microseconds>(end - start).count());
    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",\
	pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);

    int *gpuWall, *gpuResult[2];
    int size = rows*cols;

    gpuResult[0] = static_cast<int *>(mgpu::cudaMalloc(sizeof(int) * cols));
    gpuResult[1] = static_cast<int *>(mgpu::cudaMalloc(sizeof(int) * cols));
    mgpu::cudaMemcpy(gpuResult[0], ::data, sizeof(int)*cols, cudaMemcpyHostToDevice);
    gpuWall = static_cast<int *>(mgpu::cudaMalloc(sizeof(int) * (size - cols)));
    mgpu::cudaMemcpy(gpuWall, ::data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice);


    int final_ret = calc_path(gpuWall, gpuResult, rows, cols, \
	 pyramid_height, blockCols, borderCols);

    mgpu::cudaMemcpy(result, gpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost);


#ifdef BENCH_PRINT

    for (int i = 0; i < cols; i++)

        printf("%d ",::data[i]) ;

    printf("\n") ;

    for (int i = 0; i < cols; i++)

        printf("%d ",result[i]) ;

    printf("\n") ;

#endif


    mgpu::cudaFree(gpuWall);
    mgpu::cudaFree(gpuResult[0]);
    mgpu::cudaFree(gpuResult[1]);

    mgpu::cudaFreeHost(::data);
    delete [] wall;
    mgpu::cudaFreeHost(result);

}


