//Udacity HW 4
//Radix Sorting

#include "reference_calc.cpp"
#include "utils.h"

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */


__global__
void getHist(unsigned int* const d_inputVals, unsigned int* d_histo)
{
	int myiD = blockIdx.x * blockDim.x + threadId.x;
	unsigned int myItem = d_inputVals[myItem];
	// The input value will be pre-shifted so we will always check lsb
	int myBin = myItem & 1;
	atomicAdd(&(d_histo[myBin],1);
	
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{

	// ********* Step 1: Create Histogram ********** //
	
	// numBits represents the number of bits we will be scanning
	const int numBits = 1;
	// The number of Bins is equal to 1 * 2^(numBits)
	const int numBins = 1 << numBits;
	unsigned int* d_histo;

	// Allocate histogram memory on device and set to zero
	checkCudaErrors(cudaMalloc(d_histo, sizeof(int)*numBins));
	checkCudaErrors(cudaMemset(d_histo,0, sizeof(int)*numBins));

	int threads = 1024;
	int blocks  = numElems / threads;
		
	

	// **************** End Step 1 ***************** //
 
}

