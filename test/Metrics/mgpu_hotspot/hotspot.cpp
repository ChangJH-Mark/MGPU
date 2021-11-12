//
// Created by root on 2021/4/27.
//

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <time.h>
#include <assert.h>
#include "client/api.h"

#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 16
#endif

#define STR_SIZE 256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5
/* chip parameters	*/
float t_chip = 0.0005;
float chip_height = 0.016;
float chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
float amb_temp = 80.0;

void run(int argc, char** argv);

/* define timer macros */
#define pin_stats_reset()   startCycle()
#define pin_stats_pause(cycles)   stopCycle(cycles)
#define pin_stats_dump(cycles)    printf("timer: %Lu\n", cycles)



void
fatal(char *s)
{
    fprintf(stderr, "error: %s\n", s);

}

void writeoutput(float *vect, int grid_rows, int grid_cols, char *file){

    int i,j, index=0;
    FILE *fp;
    char str[STR_SIZE];

    if( (fp = fopen(file, "w" )) == 0 )
        printf( "The file was not opened\n" );


    for (i=0; i < grid_rows; i++)
        for (j=0; j < grid_cols; j++)
        {

            sprintf(str, "%d\t%g\n", index, vect[i*grid_cols+j]);
            fputs(str,fp);
            index++;
        }

    fclose(fp);
}


void readinput(float *vect, int grid_rows, int grid_cols, char *file){

    int i,j;
    FILE *fp;
    char str[STR_SIZE];
    float val;

    if( (fp  = fopen(file, "r" )) ==0 )
        printf( "The file was not opened\n" );


    for (i=0; i <= grid_rows-1; i++)
        for (j=0; j <= grid_cols-1; j++)
        {
            fgets(str, STR_SIZE, fp);
            if (feof(fp))
                fatal("not enough lines in file");
            //if ((sscanf(str, "%d%f", &index, &val) != 2) || (index != ((i-1)*(grid_cols-2)+j-1)))
            if ((sscanf(str, "%f", &val) != 1))
                fatal("invalid file format");
            vect[i*grid_cols+j] = val;
        }

    fclose(fp);

}

/*
   compute N time steps
*/
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

int compute_tran_temp(float *MatrixPower,float *MatrixTemp[2], int col, int row, \
		int total_iterations, int num_iterations, int blockCols, int blockRows, int borderCols, int borderRows)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(blockCols, blockRows);

    float grid_height = chip_height / row;
    float grid_width = chip_width / col;

    float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
    float Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
    float Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
    float Rz = t_chip / (K_SI * grid_height * grid_width);

    float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
    float step = PRECISION / max_slope;
    float t;
    float time_elapsed;
    time_elapsed=0.001;

    int src = 1, dst = 0;

    for (t = 0; t < total_iterations; t+=num_iterations) {
        int temp = src;
        src = dst;
        dst = temp;
        auto conf = mgpu::LaunchConf{dimGrid, dimBlock};
        mgpu::cudaLaunchKernel(conf,"/opt/custom/ptx/hotspot.ptx", "calculate_temp",int(MIN(num_iterations, total_iterations-t)), MatrixPower,MatrixTemp[src],MatrixTemp[dst],\
		col,row,borderCols, borderRows, Cap,Rx,Ry,Rz,step,time_elapsed);
    }
    return dst;
}

void usage(int argc, char **argv)
{
    fprintf(stderr, "Usage: %s <grid_rows/grid_cols> <pyramid_height> <sim_time> <temp_file> <power_file> <output_file>\n", argv[0]);
    fprintf(stderr, "\t<grid_rows/grid_cols>  - number of rows/cols in the grid (positive integer)\n");
    fprintf(stderr, "\t<pyramid_height> - pyramid heigh(positive integer)\n");
    fprintf(stderr, "\t<sim_time>   - number of iterations\n");
    fprintf(stderr, "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
    fprintf(stderr, "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
    fprintf(stderr, "\t<output_file> - name of the output file\n");
    exit(1);
}

int main(int argc, char** argv)
{
    auto start = std::chrono::steady_clock::now();

    run(argc,argv);
    auto end = std::chrono::steady_clock::now();
    printf("cost usec %ld \n", std::chrono::duration_cast<chrono::microseconds>(end - start).count());
    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    int size;
    int grid_rows,grid_cols;
    float *FilesavingTemp,*FilesavingPower,*MatrixOut;
    char *tfile, *pfile, *ofile;

    int total_iterations = 60;
    int pyramid_height = 1; // number of iterations

    if (argc != 7)
        usage(argc, argv);
    if((grid_rows = atoi(argv[1]))<=0||
       (grid_cols = atoi(argv[1]))<=0||
       (pyramid_height = atoi(argv[2]))<=0||
       (total_iterations = atoi(argv[3]))<=0)
        usage(argc, argv);

    tfile=argv[4];
    pfile=argv[5];
    ofile=argv[6];

    size=grid_rows*grid_cols;

    /* --------------- pyramid parameters --------------- */
# define EXPAND_RATE 2// add one iteration will extend the pyramid base by 2 per each borderline
    int borderCols = (pyramid_height)*EXPAND_RATE/2;
    int borderRows = (pyramid_height)*EXPAND_RATE/2;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int smallBlockRow = BLOCK_SIZE-(pyramid_height)*EXPAND_RATE;
    int blockCols = grid_cols/smallBlockCol+((grid_cols%smallBlockCol==0)?0:1);
    int blockRows = grid_rows/smallBlockRow+((grid_rows%smallBlockRow==0)?0:1);

    // FilesavingTemp = (float *) malloc(size*sizeof(float));
    FilesavingTemp = static_cast<float *>(mgpu::cudaMallocHost(size * sizeof(float)));
    // FilesavingPower = (float *) malloc(size*sizeof(float));
    FilesavingPower = static_cast<float *>(mgpu::cudaMallocHost(size * sizeof(float)));
    //MatrixOut = (float *) calloc (size, sizeof(float));
    MatrixOut = static_cast<float *>(mgpu::cudaMallocHost(size * sizeof(float)));

    if( !FilesavingPower || !FilesavingTemp || !MatrixOut)
        fatal("unable to allocate memory");

    printf("pyramidHeight: %d\ngridSize: [%d, %d]\nborder:[%d, %d]\nblockGrid:[%d, %d]\ntargetBlock:[%d, %d]\n",\
	pyramid_height, grid_cols, grid_rows, borderCols, borderRows, blockCols, blockRows, smallBlockCol, smallBlockRow);

    readinput(FilesavingTemp, grid_rows, grid_cols, tfile);
    readinput(FilesavingPower, grid_rows, grid_cols, pfile);

    float *MatrixTemp[2], *MatrixPower;
    MatrixTemp[0] = static_cast<float *>(mgpu::cudaMalloc(sizeof(float) * size));
    MatrixTemp[1] = static_cast<float *>(mgpu::cudaMalloc(sizeof(float) * size));
    mgpu::cudaMemcpy(MatrixTemp[0], FilesavingTemp, sizeof(float)*size, cudaMemcpyHostToDevice);

    MatrixPower = static_cast<float *>(mgpu::cudaMalloc(sizeof(float) * size));
    mgpu::cudaMemcpy(MatrixPower, FilesavingPower, sizeof(float)*size, cudaMemcpyHostToDevice);
    printf("Start computing the transient temperature\n");
    int ret = compute_tran_temp(MatrixPower,MatrixTemp,grid_cols,grid_rows, \
	 total_iterations,pyramid_height, blockCols, blockRows, borderCols, borderRows);
    printf("Ending simulation\n");
    mgpu::cudaMemcpy(MatrixOut, MatrixTemp[ret], sizeof(float)*size, cudaMemcpyDeviceToHost);

    writeoutput(MatrixOut,grid_rows, grid_cols, ofile);

    mgpu::cudaFree(MatrixPower);
    mgpu::cudaFree(MatrixTemp[0]);
    mgpu::cudaFree(MatrixTemp[1]);
    mgpu::cudaFreeHost(MatrixOut);
}
