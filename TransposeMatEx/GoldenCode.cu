#include <stdio.h>
#include "gputimer.h"

const int N = 1024;	// matrix is N*N
const int K = 32;	// each block will be 32 * 32 blocks

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
	
//	printf("input matrix:\n"); print_matrix(in);
//	printf("reference or 'gold' transposed matrix:\n"); print_matrix(gold);
	
	
}
