#include "./GPU_Atomics.cuh"

__global__ void kernel_Atomics()
{
	
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
//	return 0;
}

int call_AtomicsOnGPU()
{
	// Start same as CPU - creating databuffer
	unsigned char* dataBuffer = (unsigned char*)big_random_block(SIZE);

	// CUDA Events to record timing - initialize
	cudaEvent_t start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// Memory allocation on the GPU for the input data
	unsigned char* device_dataBuffer;
	unsigned int* device_mainHistogram;
	HANDLE_ERROR(cudaMalloc((void**)&device_dataBuffer, SIZE));		// Allocate memory for data
	HANDLE_ERROR(cudaMemcpy(device_dataBuffer, dataBuffer, SIZE, cudaMemcpyHostToDevice));		// Copy data from CPU to GPU
	HANDLE_ERROR(cudaMalloc((void**)&device_mainHistogram, 256*sizeof(long)));		// Allocate memory for histogram bins

	// cudaMemset = same as the C memset - used to fill a block of memory with a particular value
	HANDLE_ERROR(cudaMemset(device_mainHistogram, 0, 256 * sizeof(int)));
}

