#include "./GPU_Atomics.cuh"


// Main kernel that computes the histogram
// Args: (pointer to input data array, length of the input array, pointer to the output histogram buffer
// Threads start at offsett b/w 0 and go up to num of threads-1 and the stride is by total number of threads launched
__global__ void kernel_Atomics(unsigned char *device_inputBuffer, long device_inputBufferSize, unsigned int* device_outputBuffer)
{
	// Allocating a shared memory buffer to hold intermmediate histogram - basically each one of the 256 elements will be calculated by the constituent threads of a thread block. Syncing will be required
	// Without using a limited shared memory buffer, thousands of threads will be trying to access a small number of memory addresses
	__shared__ unsigned int device_tempBuffer[256];
	device_tempBuffer[threadIdx.x] = 0;		// Initializing the temp buffer to hold 0 as the start value
	__syncthreads();	// Ensures that every thread has completed its write operation before progressing

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	// Walk through the i/p array and then update corresponding device histogram bin
	while (i < device_inputBufferSize)
	{
		atomicAdd(&device_tempBuffer[device_inputBuffer[i]], 1);
		i += offset;
	}

	__syncthreads();

	// Merge all individual temporary histograms produced by each thread block into the final buffer
	// Here we use 256 threads and have 256 bins - thus each thread automatically adds a single bin to the final buffer - order in which the blocks add are random, but since addition is commutative, final answer will be the same always
	atomicAdd(&(device_outputBuffer[threadIdx.x]), device_tempBuffer[threadIdx.x]);
}

void main_Atomics()
{
}

int call_Atomics()
{
//	// Create a random stream of data that is 100 MB (defined by SIZE)
//	unsigned char* dataBuffer = (unsigned char*)big_random_block(SIZE);
//
//	// Create bin array for storing 8-bit byte value (8 bit byte can be anything of 256 values i.e. 0 to 255) at each index
//	unsigned int mainHistogram[256];
//	for (int i = 0; i < 256; i++)
//	{
//		mainHistogram[i] = 0;	// Initialize with 0
//	}
//
//	// LOGIC: If dataBuffer[i] is whats looked at, we need to increment the bin with that value's in the main histogram, like a counter. Location of dataBuffer[i] is given by mainHistogram[dataBuffer[i]]
//	// E.g. if dataBuffer[i] = 3, then the bin with index = value i.e. index of 3, of the histogram, is a counter for how many times the value of 3 is seen. The bin with index 3 (i.e. the 4th bin since the counting starts at 0) is in location of mainHistogram[3]. We are matching up the values with the indices basically.
//	for (int i = 0; i < SIZE; i++)
//	{
//		mainHistogram[dataBuffer[i]]++;
//	}
//
//	// Verifying the data is corrected by summing all values in bin - sum is always the same regardless of random array
//	long histogramCount = 0;
//	for (int i = 0; i < 256; i++)
//	{
//		histogramCount += mainHistogram[i];
//	}
//
//	printf("Sum: %1d\n", histogramCount);
//	
//	free(dataBuffer);
	return 0;
}

int call_AtomicsOnGPU()
{
	// Start same as CPU - creating databuffer
	unsigned char* host_dataBuffer = (unsigned char*)big_random_block(SIZE);

	// CUDA Events to record timing - initialize
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// Memory allocation on the GPU for the input data
	unsigned char* device_dataBuffer;
	unsigned int* device_mainHistogram;
	HANDLE_ERROR(cudaMalloc((void**)&device_dataBuffer, SIZE));		// Allocate memory for data
	HANDLE_ERROR(cudaMemcpy(device_dataBuffer, host_dataBuffer, SIZE, cudaMemcpyHostToDevice));		// Copy data from CPU to GPU
	HANDLE_ERROR(cudaMalloc((void**)&device_mainHistogram, 256*sizeof(long)));		// Allocate memory for histogram bins

	// cudaMemset = same as the C memset - used to fill a block of memory with a particular value
	HANDLE_ERROR(cudaMemset(device_mainHistogram, 0, 256 * sizeof(int)));

	// Computing the histogram stage
	// Figuring out the device properties
	cudaDeviceProp device_prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&device_prop, 0));
	int blocks = device_prop.multiProcessorCount;
	// The blocks*2 value comes from trial and error in the book as the most optimal
	// Usually we need to think here, if we have a 100mb data i.e. 104,857,600 bytes, and we divide by 256, we get 409,600 blocks - which might not always be the best due to the way GPUs work
	kernel_Atomics << <blocks * 2, 256 >> > (device_dataBuffer, SIZE, device_mainHistogram);


	// Copying the histogram back into the CPU from GPU
	unsigned int host_mainHistogram[256];
	HANDLE_ERROR(cudaMemcpy(host_mainHistogram, device_mainHistogram, 256 * sizeof(int), cudaMemcpyDeviceToHost));

	// Stop timers
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time to generate is %3.1f ms\n", elapsedTime);

	// Verification that the histogram has been calculated correctly by the GPU
	long host_histoCount = 0;
	for (int i = 0; i < 256; i++)
	{
		host_histoCount += host_mainHistogram[i];
	}
	printf("Histogram sum is %1d\n", host_histoCount);

	// Verification with CPU generated histogram (in reverse) i.e. when CPU sees data elements, it removes it, so that the overall sum at the end is 0
	for (int i = 0; i < SIZE; i++)
	{
		host_mainHistogram[host_dataBuffer[i]]--;
	}
	for (int i = 0; i < 256; i++)
	{
		if (host_mainHistogram[i] != 0)
		{
			printf("Failure at %d!\n", i);
		}
	}

	// Cleaning up CUDA events, GPU memory, and CPU memory
	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	cudaFree(device_mainHistogram);
	cudaFree(device_dataBuffer);
	free(host_dataBuffer);

	return 0;
}

