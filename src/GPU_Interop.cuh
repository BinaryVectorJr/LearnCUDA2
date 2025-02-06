#define GL_GLEXT_PROTOTYPES

#include "./CUDA_Helpers.cuh"
#include "./GL_Helper.cuh"

#define DIM 512

// Kernel that actually does the GPU calculation (the one that is used with the triple angle brackets)
__global__ void kernel_Interop(uchar4* ptr);

// Main program doing the memory allocation, transfer, and freeing as well as calling the kernel
// Has the same args as the kernel call above
void main_Interop();

// Call program that should be put in the "main" of the solution CPP
int call_Interop(int argc, char** argv);

static void draw_func(void);

static void key_func(unsigned char key, int x, int y);

