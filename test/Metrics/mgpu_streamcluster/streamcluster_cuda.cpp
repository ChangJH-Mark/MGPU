/***********************************************
	streamcluster_cuda.cu
	: parallelized code of streamcluster

	- original code from PARSEC Benchmark Suite
	- parallelization with CUDA API has been applied by

	Shawn Sang-Ha Lee - sl4ge@virginia.edu
	University of Virginia
	Department of Electrical and Computer Engineering
	Department of Computer Science

***********************************************/
#include "streamcluster_header.cu"
#include "client/api.h"

using namespace std;
using namespace mgpu;

// AUTO-ERROR CHECK FOR ALL CUDA FUNCTIONS
#define CUDA_SAFE_CALL( call) do {										\
   cudaError err = call;												\
   if( cudaSuccess != err) {											\
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
               __FILE__, __LINE__, cudaGetErrorString( err) );			\
   exit(EXIT_FAILURE);													\
   } } while (0)

#define THREADS_PER_BLOCK 512
#define MAXBLOCKS 65536
#define CUDATIME

// host memory
float *work_mem_h;
float *coord_h;

// device memory
float *work_mem_d;
float *coord_d;
int   *center_table_d;
bool  *switch_membership_d;
Point *p;

static int iter = 0;		// counter for total# of iteration


//=======================================
// Allocate Device Memory
//=======================================
void allocDevMem(int num, int dim)
{
    center_table_d = static_cast<int *>(mgpu::cudaMalloc(num * sizeof(int)));
    switch_membership_d = static_cast<bool *>(mgpu::cudaMalloc(num * sizeof(bool)));
    p = static_cast<Point *>(mgpu::cudaMalloc(num * sizeof(Point)));
    coord_d = static_cast<float *>(mgpu::cudaMalloc(num * dim * sizeof(float)));
}

//=======================================
// Allocate Host Memory
//=======================================
void allocHostMem(int num, int dim)
{
    coord_h = static_cast<float *>(mgpu::cudaMallocHost(num * dim * sizeof(float)));
}

//=======================================
// Free Device Memory
//=======================================
void freeDevMem()
{
    mgpu::cudaFree(center_table_d);
    mgpu::cudaFree(switch_membership_d);
    mgpu::cudaFree(p);
    mgpu::cudaFree(coord_d);
}

//=======================================
// Free Host Memory
//=======================================
void freeHostMem()
{
    mgpu::cudaFreeHost(coord_h);
}

//=======================================
// pgain Entry - CUDA SETUP + CUDA CALL
//=======================================
float pgain( long x, Points *points, float z, long int *numcenters, int kmax, bool *is_center, int *center_table, bool *switch_membership, bool isCoordChanged,
             double *serial_t, double *cpu_to_gpu_t, double *gpu_to_cpu_t, double *alloc_t, double *kernel_t, double *free_t)
{
#ifdef CUDATIME
    float tmp_t;
    mgpu::event_t start, stop;
    mgpu::cudaEventCreate(&start);
    mgpu::cudaEventCreate(&stop);

    mgpu::cudaEventRecord(start, 0);
#endif

    cudaError_t error;

    int stride	= *numcenters + 1;			// size of each work_mem segment
    int K		= *numcenters ;				// number of centers
    int num		=  points->num;				// number of points
    int dim		=  points->dim;				// number of dimension
    int nThread =  num;						// number of threads == number of data points

    //=========================================
    // ALLOCATE HOST MEMORY + DATA PREPARATION
    //=========================================
    work_mem_h = static_cast<float *>(mgpu::cudaMallocHost(stride * (nThread + 1) * sizeof(float)));
    // Only on the first iteration
    if(iter == 0)
    {
        allocHostMem(num, dim);
    }

    // build center-index table
    int count = 0;
    for( int i=0; i<num; i++)
    {
        if( is_center[i] )
        {
            center_table[i] = count++;
        }
    }

    // Extract 'coord'
    // Only if first iteration OR coord has changed
    if(isCoordChanged || iter == 0)
    {
        for(int i=0; i<dim; i++)
        {
            for(int j=0; j<num; j++)
            {
                coord_h[ (num*i)+j ] = points->p[j].coord[i];
            }
        }
    }

#ifdef CUDATIME
    mgpu::cudaEventRecord(stop,0);
    mgpu::cudaEventSynchronize(stop);
    mgpu::cudaEventElapsedTime(&tmp_t, start, stop);
    *serial_t += (double) tmp_t;

    mgpu::cudaEventRecord(start,0);
#endif

    //=======================================
    // ALLOCATE GPU MEMORY
    //=======================================
    work_mem_d = static_cast<float *>(mgpu::cudaMalloc(stride * (nThread + 1) * sizeof(float)));
    // Only on the first iteration
    if( iter == 0 )
    {
        allocDevMem(num, dim);
    }

#ifdef CUDATIME
    mgpu::cudaEventRecord(stop,0);
    mgpu::cudaEventSynchronize(stop);
    mgpu::cudaEventElapsedTime(&tmp_t, start, stop);
    *alloc_t += (double) tmp_t;

    mgpu::cudaEventRecord(start,0);
#endif

    //=======================================
    // CPU-TO-GPU MEMORY COPY
    //=======================================
    // Only if first iteration OR coord has changed
    if(isCoordChanged || iter == 0)
    {
        mgpu::cudaMemcpy(coord_d,  coord_h,	 num * dim * sizeof(float), cudaMemcpyHostToDevice);
    }
    mgpu::cudaMemcpy(center_table_d,  center_table,  num * sizeof(int),   cudaMemcpyHostToDevice);
    mgpu::cudaMemcpy(p,  points->p,				   num * sizeof(Point), cudaMemcpyHostToDevice);

    mgpu::cudaMemset((void*) switch_membership_d, 0,			num * sizeof(bool));
    mgpu::cudaMemset((void*) work_mem_d,  		0, stride * (nThread + 1) * sizeof(float));

#ifdef CUDATIME
    mgpu::cudaEventRecord(stop,0);
    mgpu::cudaEventSynchronize(stop);
    mgpu::cudaEventElapsedTime(&tmp_t, start, stop);
    *cpu_to_gpu_t += (double) tmp_t;

    mgpu::cudaEventRecord(start,0);
#endif

    //=======================================
    // KERNEL: CALCULATE COST
    //=======================================
    // Determine the number of thread blocks in the x- and y-dimension
    int num_blocks 	 = (int) ((float) (num + THREADS_PER_BLOCK - 1) / (float) THREADS_PER_BLOCK);
    int num_blocks_y = (int) ((float) (num_blocks + MAXBLOCKS - 1)  / (float) MAXBLOCKS);
    int num_blocks_x = (int) ((float) (num_blocks+num_blocks_y - 1) / (float) num_blocks_y);
    dim3 grid_size(num_blocks_x, num_blocks_y, 1);
    mgpu::config conf;
    conf.grid = grid_size;
    conf.block = THREADS_PER_BLOCK;
    mgpu::cudaLaunchKernel(conf, "/opt/custom/ptx/streamcluster.ptx", "kernel_compute_cost",
                           num,					// in:	# of data
                           dim,					// in:	dimension of point coordinates
                           x,						// in:	point to open a center at
                           p,						// in:	data point array
                           K,						// in:	number of centers
                           stride,					// in:  size of each work_mem segment
                           coord_d,				// in:	array of point coordinates
                           work_mem_d,				// out:	cost and lower field array
                           center_table_d,			// in:	center index table
                           switch_membership_d		// out:  changes in membership
                           );
    mgpu::cudaStreamSynchronize(0);

    // error check
//    error = cudaGetLastError();
//    if (error != cudaSuccess)
//    {
//        printf("kernel error: %s\n", cudaGetErrorString(error));
//        exit(EXIT_FAILURE);
//    }

#ifdef CUDATIME
    mgpu::cudaEventRecord(stop,0);
    mgpu::cudaEventSynchronize(stop);
    mgpu::cudaEventElapsedTime(&tmp_t, start, stop);
    *kernel_t += (double) tmp_t;

    mgpu::cudaEventRecord(start,0);
#endif

    //=======================================
    // GPU-TO-CPU MEMORY COPY
    //=======================================
    mgpu::cudaMemcpy(work_mem_h, 		  work_mem_d, 	stride * (nThread + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    mgpu::cudaMemcpy(switch_membership, switch_membership_d,	 num * sizeof(bool),  cudaMemcpyDeviceToHost);

#ifdef CUDATIME
    mgpu::cudaEventRecord(stop,0);
    mgpu::cudaEventSynchronize(stop);
    mgpu::cudaEventElapsedTime(&tmp_t, start, stop);
    *gpu_to_cpu_t += (double) tmp_t;

    mgpu::cudaEventRecord(start,0);
#endif

    //=======================================
    // CPU (SERIAL) WORK
    //=======================================
    int number_of_centers_to_close = 0;
    float gl_cost_of_opening_x = z;
    float *gl_lower = &work_mem_h[stride * nThread];
    // compute the number of centers to close if we are to open i
    for(int i=0; i < num; i++)
    {
        if( is_center[i] )
        {
            float low = z;
            for( int j = 0; j < num; j++ )
            {
                low += work_mem_h[ j*stride + center_table[i] ];
            }

            gl_lower[center_table[i]] = low;

            if ( low > 0 )
            {
                ++number_of_centers_to_close;
                work_mem_h[i*stride+K] -= low;
            }
        }
        gl_cost_of_opening_x += work_mem_h[i*stride+K];
    }

    //if opening a center at x saves cost (i.e. cost is negative) do so; otherwise, do nothing
    if ( gl_cost_of_opening_x < 0 )
    {
        for(int i = 0; i < num; i++)
        {
            bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
            if ( switch_membership[i] || close_center )
            {
                points->p[i].cost = dist(points->p[i], points->p[x], dim) * points->p[i].weight;
                points->p[i].assign = x;
            }
        }

        for(int i = 0; i < num; i++)
        {
            if( is_center[i] && gl_lower[center_table[i]] > 0 )
            {
                is_center[i] = false;
            }
        }

        if( x >= 0 && x < num)
        {
            is_center[x] = true;
        }
        *numcenters = *numcenters + 1 - number_of_centers_to_close;
    }
    else
    {
        gl_cost_of_opening_x = 0;
    }

    //=======================================
    // DEALLOCATE HOST MEMORY
    //=======================================
    mgpu::cudaFreeHost(work_mem_h);


#ifdef CUDATIME
    mgpu::cudaEventRecord(stop,0);
    mgpu::cudaEventSynchronize(stop);
    mgpu::cudaEventElapsedTime(&tmp_t, start, stop);
    *serial_t += (double) tmp_t;

    mgpu::cudaEventRecord(start,0);
#endif

    //=======================================
    // DEALLOCATE GPU MEMORY
    //=======================================
    mgpu::cudaFree(work_mem_d);


#ifdef CUDATIME
    mgpu::cudaEventRecord(stop,0);
    mgpu::cudaEventSynchronize(stop);
    mgpu::cudaEventElapsedTime(&tmp_t, start, stop);
    *free_t += (double) tmp_t;
#endif
    iter++;
    return -gl_cost_of_opening_x;
}
