#include "./CUDA_Helpers.cuh"

// Kernel that actually does the GPU calculation (the one that is used with the triple angle brackets)
__global__ void kernel_VectorAdd(double* A, double* B, double* C, int arraySize);

// Main program doing the memory allocation, transfer, and freeing as well as calling the kernel
void main_VectorAdd(double* A, double* B, double* C, int arraySize);

// Call program that should be put in the "main" of the solution CPP
void call_VectorAdd();