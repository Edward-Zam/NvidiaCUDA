#include <stdio.h>
#include "gputimer.h"

const int N = 1024;	// matrix is N*N
const int K = 16;	// each block will be 32 * 32 blocks

// Utility functions: compare, print, and fill matrices
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at: %s : %d\n", file,line);
    fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);;
    exit(1);
  }
}

int compare_matrices(float *gpu, float *ref)
{
	int result = 0;

	for(int j=0; j < N; j++)
    	for(int i=0; i < N; i++)
    		if (ref[i + j*N] != gpu[i + j*N])
    		{
    			// printf("reference(%d,%d) = %f but test(%d,%d) = %f\n",
    			// i,j,ref[i+j*N],i,j,test[i+j*N]);
    			result = 1;
    		}
    return result;
}

void print_matrix(float *mat)
{
	for(int j=0; j < N; j++) 
	{
		for(int i=0; i < N; i++) { printf("%4.4g ", mat[i + j*N]); }
		printf("\n");
	}	
}

// fill a matrix with sequential numbers in the range 0..N-1
void fill_matrix(float *mat)
{
	for(int j=0; j < N * N; j++)
		mat[j] = (float) j;
}

void transpose_CPU(float in [], float out[])
{
	for(int j=0;j<N;j++)
		for(int i=0;i<N;i++)
			out[j + i*N] = in[i + j*N]; //out(j,i) = in(i,j)
}

__global__ void 
transpose_serial(float in [], float out[])
{
	for(int j=0;j<N;j++)
		for(int i=0;i<N;i++)
			out[j + i*N] = in[i + j*N]; //out(j,i) = in(i,j)
}

__global__ void
transpose_parallel_per_row(float in[], float out[])
{
	int i = threadIdx.x;
	
	for(int j=0; j<N;j++)
		out[i + i*N] = in[i + j*N];
}

__global__ void
transpose_parallel_per_element(float in[], float out[])
{
	int i	= blockIdx.x * blockDim.x + threadIdx.x;
	int j	= blockIdx.y * blockDim.y + threadIdx.y;
	
	out[j + i*N] = in[i + j*N];
}

// to be launched with one thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elmts
__global__ void 
transpose_parallel_per_element_tiled(float in[], float out[])
{	
	// corner in and corner out are transposed
	int corner_i_in		= blockIdx.x * K, corner_j_in  = blockIdx.y * K;
	int corner_i_out	= blockIdx.y * K, corner_j_out = blockIdx.x * K;
	
	int x = threadIdx.x, y = threadIdx.y;
	
	__shared__ float tile[K][K];
	
	// coalesced read from global memory, transposed write to shared mem
	tile[y][x] = in[(corner_i_in + x)+ (corner_j_in + y) * N];
	__syncthreads();
	
	// read from shared memory, coalesced write
	out[(corner_i_out + x) + (corner_j_out + y) * N] = tile[x][y];
	
}

int main(int argc, char **argv)
{
	int numbytes = N*N*sizeof(float);
	
	float *in	= (float *) malloc(numbytes);
	float *out	= (float *) malloc(numbytes);
	float *gold	= (float *) malloc(numbytes);
	
	fill_matrix(in);
	transpose_CPU(in, gold);
	
	float *d_in, *d_out;
	
	cudaMalloc(&d_in, numbytes);
	cudaMalloc(&d_out, numbytes);
	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);
	
	GpuTimer timer;
	timer.Start();
	transpose_serial<<<1,1>>>(d_in, d_out); //takes about 0.084768 ms for N = 8; 
	// transpose_serial: 466.429 ms for N = 1024;
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_serial: %g ms. \nVerifying transpose...%s\n,", 
		timer.Elapsed(),
		compare_matrices(out,gold) ? "Failed": "Success"); 
		
	timer.Start();	
	transpose_parallel_per_row<<<1,N>>>(d_in, d_out);
	// transpose_parallel_per_row: 4.76848 ms
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_row: %g ms. \nVerifying transpose...%s\n,", 
		timer.Elapsed(),
		compare_matrices(out,gold) ? "Failed": "Success"); 
		
	dim3 blocks(N/K, N/K);
	dim3 threads(K,K);
	timer.Start();
	transpose_parallel_per_element<<<blocks, threads>>>(d_in, d_out);
	// transpose_parallel_per_element: 0.341504 ms for N=1024
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element: %g ms. \nVerifying tranpose...%s\n,",
		timer.Elapsed(),
		compare_matrices(out,gold) ? "Failed": "Success");
		
	timer.Start();
	transpose_parallel_per_element_tiled<<<blocks,threads>>>(d_in,d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element_tiled: %g ms. \nVerifying transpose...%s\n,", 
	timer.Elapsed(),
	compare_matrices(out,gold)?"Failed":"Success");
	
//	printf("input matrix:\n"); print_matrix(in);
//	printf("reference or 'gold' transposed matrix:\n"); print_matrix(gold);
	
	
}
