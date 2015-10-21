//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"

const int threads = 1024;	// number of threads in block
const int twidth = 32;		// tile size is twdith X twidth

__global__
void createMask( const uchar4* const d_sourceImg, unsigned char* const mask,
				const size_t numRowsSource, const size_t numColsSource)
{
	// Find global ID from an K*K block
	//id = blockIdx.x * blockIdx.y + threadIdx.x;
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadId.x,
	blockIdx.y * blockDim.y + threadIdx.y);
	
	const int gid = thread_2D_pos.y * numColsSource + thread_2D_pos.x;
	
	int tid = blockDim.x * threadIdx.y + threadId.x;
	
	if( thread_2D_pos.x >= numColsSource || thread_2D_pos.y >= numRowsSource)
	{
		return; // return if trying to access memory outside of GPU memory.
	}
	
	mask[tid] = (d_sourceImg[gid].x + d_sourceImg[gid].y + d_sourceImg[gid].z < 3*255) ? 1:0;
}

__global__
void sortPixels()
{
	// Find global ID from an K*K block
	//id = blockIdx.x * blockIdx.y + threadIdx.x;
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadId.x,
	blockIdx.y * blockDim.y + threadIdx.y);
	
	const int gid = thread_2D_pos.y * numColsSource + thread_2D_pos.x;
	
	int tid = blockDim.x * threadIdx.y + threadId.x;
	
	if( thread_2D_pos.x >= numColsSource || thread_2D_pos.y >= numRowsSource)
	{
		return; // return if trying to access memory outside of GPU memory.
	}
	
	// grab neighboring pixels from mask
	char right	= mask[gid + 1];
	char left	= mask[gid - 1];
	char top	= mask[gid -numRowsSource];
	char bottom = mask[gid + numRowSource];
	
	// check each pixel and its neighbors. 
	// Don't check borders
	if( (thread_2D_pos.y != 0) && (thread_2D_pos.y != numRowsSource -1) &&
		(thread_2D_pos.x != 0) && (thread_2D_pos.x != numColsSource -1))
	{
		if (mask[gid])
		{
			if(mask(right) && mask(left) && mask(top) && mask(bottom)
			{
				interiorPixels[tid]	= 1;
				borderPixels[tid]	= 0;
			}
			else
			{
				interiorPixels[tid]	= 0;
				borderPixels[tid]	= 1;
			}
		}
		else
		{
			interiorPixels[tid] = 0;
			borderPixels[tid]	= 0;
		}
	}
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.
	*/
	const size_t srcSize = numRowsSource * numColsSource;
	unsigned char* d_mask;
	unsigned char* d_borderPixels;
	unsigned char* d_interiorPixels;
	
	// Initialize threads to max allowed per block for this GPU.
	dim3 threads(twidth, twidth);
	// Calculate number of blocks
	dim3 blocks(numColsSource/twidth +1, numRowsSource/twidth +1);
	
	// Allocate device memory
	checkCudaErrors(cudaMalloc(&d_sourceImg, sizeof(char4) * srcSize));
	checkCudaErrors(cudaMalloc(&d_mask, sizeof(uchar) * srcSize));
	checkCudaErrors(cudaMalloc(&d_borderPixels, sizeof(uchar) * srcSize));
	checkCudaErrors(cudaMalloc(&d_interiorPixels, sizeof(uchar) * srcSize));
	
	
	// Copy h_source to d_source
	checkCudaErrors(cudaMemppy(d_sourceImg, h_sourceImg, sizeof(char4) * srcSize, cudaMemcpyHostToDevice));
	
	// Launch kernel to create mask
	createMask<<<blocks,threads>>>(d_sourceImg, mask);
	
	/*
     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.
	*/
	// Launch kernel to computer interior and border pixels.
	// threads and blocks remain the same since we are still working with srcSize
	
	/*
     3) Separate out the incoming image into three separate channels
     */
	/*
     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.
	*/
	/* 
     5) For each color channel perform the Jacobi iteration described 
        above 800 times.
	*/
	/*
     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.
        */
     /*

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
      */
      
  



  /* The reference calculation is provided below, feel free to use it
     for debugging purposes. 
   */

  /*
    uchar4* h_reference = new uchar4[srcSize];
    reference_calc(h_sourceImg, numRowsSource, numColsSource,
                   h_destImg, h_reference);

    checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * srcSize, 2, .01);
    delete[] h_reference; */
}
