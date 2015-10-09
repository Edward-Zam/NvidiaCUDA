/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


#include "reference_calc.cpp"
#include "utils.h"

__global__
void getMax(const float* const d_logLuminance, float* const d_bout, const size_t numels )
{
	extern __shared__ float sdata[];
	int id	= blockDim.x * blockIdx.x * threadIdx.x;
	int tid	= threadIdx.x; 
	sdata[tid] = d_logLuminance[id];
	__syncthreads();
	
	for ( int  i = 0; i <= numels - 1; i++)
	{
		id = threadId.x;
		stride = 1;
		offset = 2;
		if((threadIdx.x + 1)%offset == 0)
		{
			sdata[id]	= sdata[tid] > sdata[tid-stride] ? sdata[tid]:sdata[tid-stride];
			//sdata[tid] = max(sdata[tid], sdata[tid-stride];
		}
		
		__syncthreads();
		stride = stride*2;
		offset = offset^2;
	}
	
	if (tid == numels)
	{
		d_bout[blockIdx.x] = sdata[numels];
	}
}

__global__
void getMin(const float* const d_logLuminance, float* const d_bout, const size_t numels )
{
	extern __shared__ float sdata[];
	int id	= blockDim.x * blockIdx.x * threadIdx.x;
	int tid	= threadIdx.x; 
	sdata[tid] = d_logLuminance[id];
	__syncthreads();
	
	for ( int  i = 0; i <= numels - 1; i++)
	{
		id = threadId.x;
		stride = 1;
		offset = 2;
		if((threadIdx.x + 1)%offset == 0)
		{
			sdata[id]	= sdata[tid] > sdata[tid-stride] ? sdata[tid-stride]:sdata[tid];
			//sdata[tid] = min(sdata[tid], sdata[tid-stride];
		}
		
		__syncthreads();
		stride = stride*2;
		offset = offset^2;
	}
	
	if (tid == numels)
	{
		d_bout[blockIdx.x] = sdata[numels];
	}
}

__global__
void getHist(int *d_bins, const int *d_in, const float minLum, const float lumRange, const int BIN_COUNT)
{
	// accumulate using atomics to avoid race conditions
	int myId	= threadIdx.x + blockDim.x * blockIdx.x;
	int myItem	= d_in[myId];
	int myBin	= (myItem - minLum) / lumRange * BIN_COUNT;
	atomicAdd(&(d_bins[myBin]),1);
}

__global__
void HillisSteeleScan()
{
	// Copy bins to local memory
	extern __shared__ unsigned int t_in[];
	
	int id	= blockIdx.x * blockDim.x + threadIdx.x;
	int tid	= threadIdx.x;
	
	t_in[tid] = d_in[id];
	__syncthreads();
	
	for(unsigned int j = 1; j < BIN_COUNT; j<<1)
	{
		if(tid+j<blockDim.x)
		{
			atomicAdd(t_in[tid+j], t_in[tid]);
			__syncthreads();
		}
		d_out[id] = tid>0?t_in[tid-1]:0;
	}
	
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
	   
	float *d_temp;
	float *d_luminanceOut;
	float lumRange;
	
	unsigned int *d_bins;
	
	// Calculate number of elements in the array;
	const size_t numels = numCols * numRows;
	// Instatiate threads to max allowed per block for this GPU
	int threads = 1024;
	// Calculate number of blocks
	int blocks = numels/threads;

	// Allocate device memory
	checkCudaErrors(cudaMalloc(&d_temp, sizeof(float)*blocks))
	checkCudaErrors(cudaMalloc(&d_logLuminance, sizeof(float)*1));
	// Set memory to zero
	//checkCudaErrors(cudaMemset())
	
	// Find the minimum value in the input logLuminance channel
	getMin<<<blocks, threads, threads * sizeof(float)>>>(d_logLuminance, d_temp, numels);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	threads = blocks;
	blocks = 1;
	getMin<<<blocks, threads,threads * sizeof(float)>>>(d_temp,d_luminanceOut,threads);
	cudaDeviceSynchronize();checkCudaErrors(cudaGetLastError());
	// Copy minimum luminance value to min_logLum in host memory
	checkCudaErrors(cudaMemcpy(&min_logLum, d_luminanceOut, sizeof(float), cudaMemcpyDeviceToHost));
	
	// Reset thread and block values
	threads = 1024;
	blocks = numels/threads;
	// Find the maximum value in the input logLuminance channel
	getMax<<<blocks, threads, threads * sizeof(float)>>>(d_logLuminance, d_temp, numels);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	threads = blocks;
	blocks = 1;
	getMax<<<blocks, threads, threads * sizeof(float)>>>(d_temp, d_luminanceOut, threads);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	// Copy maximum luminance value to max_logLum in host memory
	checkCudaErrors(cudaMemcpy(&max_logLum, d_luminanceOut, sizeof(float), cudaMemcpyDeviceToHost));
	
	// Calculate the luminance range
	lumRange = max_logLum - min_logLum;
	
	// Genereate a histogram of all values in the logLuminance channel
	threads = 1024;
	blocks = numels/threads;
	getHist<<<blocks, threads>>>(d_bins, d_logLuminance, min_logLum, lumRange, numBins);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	// Scan algorithm 
	// We are to perform an exclusive scan on the histogram; the histogram has numBins elements 
	// and therefore can be launched as one block with numBins threads. 
	threads = numBins;
	blocks = 1;
	
	
	
	
}
