#include "./CUDA_Helpers.cuh"

#define SIZE (100*1024*1024)

// Kernel that actually does the GPU calculation (the one that is used with the triple angle brackets)
__global__ void kernel_Atomics(unsigned char* device_input_buffer, long device_input_buffer_size, unsigned int* device_output_buffer);

// Main program doing the memory allocation, transfer, and freeing as well as calling the kernel
// Has the same args as the kernel call above
void main_Atomics();

// Call program that should be put in the "main" of the solution CPP
int call_Atomics();

// Call for GPU version of program
int call_AtomicsOnGPU();
