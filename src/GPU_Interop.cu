#include "./GPU_Interop.cuh"

// Creating main global variables for both APIs
GLuint glBufferObj;
cudaGraphicsResource* cudaResource;

// Non Animated image - calculations are simpe as they are not time dependent and straight lines
__global__ void kernel_Interop(uchar4 *ptr)
{
	// Map from threadIdx or blockIdx to the pixel position
	// If we swap blockDim to gridDim here, the pattern changes and it looks broken but cool
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// Calculating the value at the position of x and y
	float fx = x / (float)DIM - 0.5f;
	float fy = y / (float)DIM - 0.5f;

	// Calc. color value at the (x,y) locaiton
	unsigned char green = 128 + 127 * sin(abs(fx * 100) - abs(fy * 100));

	// Access uchar4 and unsigned char*
	ptr[offset].x = 0;
	ptr[offset].y = green;
	ptr[offset].z = 0;
	ptr[offset].w = 255;
}

// Animated Ripple - calculations are different as they are time dependent and are circular ripples
__global__ void kernel_Interop_Anim(uchar4* ptr, int ticks)
{
	// Map from threadIdx or blockIdx to the pixel position
	// If we swap blockDim to gridDim here, the pattern changes and it looks broken but cool
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	// Calculating the value at the position of x and y
	float fx = x - DIM/2;
	float fy = y - DIM/2;
	float d = sqrtf(fx * fx + fy * fy);

	// Calc. color value at the (x,y) locaiton
	unsigned char green = 128 + 127 * sin(abs(fx * 100) - abs(fy * 100));

	//Alt color
	unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d/10.0f - ticks/7.0f)/(d/10.0f + 1.0f));

	ptr[offset].x = grey;
	ptr[offset].y = grey;
	ptr[offset].z = grey;
	ptr[offset].w = 255;
}

void generate_frame(uchar4* pixels, void*, int ticks)
{
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16,16);
	kernel_Interop_Anim << <grids, threads >> > (pixels, ticks);
}

void main_Interop()
{

}

// Call for animated ripple on GPU
void Interop_Anim()
{
	// Create GPU bitmap structure
	// The entire code below this has been moved to the header, so that the animations can work
	GPUAnimBitmap bitmap(DIM, DIM, NULL);
	bitmap.anim_and_exit((void(*)(uchar4*, void*, int))generate_frame, NULL);
}

// Call for static image on GPU
int call_Interop(int argc, char** argv)
{
	 // CPU Bitmap code
	 // Selecting a CUDA device to run the applicaiton on
	 cudaDeviceProp prop;
	 int device_ref;

	 // memset(void* ptr, int x, size_t n) is used to fill a block of memory with a particular value
	 // &prop = starting address of memory to be filled
	 // x = value to be filled
	 // n = number of bytes to be filled starting from &prop address
	 memset(&prop, 0, sizeof(cudaDeviceProp));

	 // Set GPU compute capability (1.0 or better)
	 prop.major = 1;
	 prop.minor = 0;
	 HANDLE_ERROR(cudaChooseDevice(&device_ref, &prop));

	// GLUT calls that need to be made before any other calls
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("bitmap");

	 // Initialize GLEW to manage OpenGL extensions and use OpenGL function
	 glewInit();

	 // device_ref also is used to know the CUDA Device ID so that we can tell the runtime to use CUDA with OpenGL
	 // This call ensures that CUDA runtime will work correctly with OpenGL driver of the GPU
	 // Must use this after GLUT (FreeGLUT) has been initialized
	 HANDLE_ERROR(cudaGLSetGLDevice(device_ref));

	// IMP: Interop operations are highly dependent on shared data buffers, which are the key components to OpenGL rendering
	// Passing data b/w OpenGL and CUDA, first step is to create buffers for both APIs
	// Step 1: Generate buffer
	glGenBuffers(1, &glBufferObj);
	// Step 2: Bind the handle to a pixel buffer
	// Basically cudaResource is a handle that is used to refer to the actual GL buffer by CUDA runtime
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, glBufferObj);
	// Step 3: Request OpenGL driver to allocate a buffer so that we can use it; GL_DYNAMIC_DRAW_ARB tells OpenGL that it will be modified repeatedly by the program; NULL means we have no data to pass to the buffer
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, DIM * DIM * 4, NULL, GL_DYNAMIC_DRAW_ARB);

	// Sharing (copying) from the OpenGL buffer to the CUDA runtime buffer
	// cudaGraphicsMapFlagsNone specifies that there is no particular behavior of this buffer that we want to specify, although we have the option to specify with cudaGraphicsMapFlagsReadOnly that the buffer will be readonly. We could also use cudaGraphicsMapFlagsWriteDiscard to specify that the previous contents will be discarded, making the buffer essentially write-only. These flags allow the CUDA and OpenGL drivers to optimize the hardware settings for buffers with restricted access patterns, although they are not required to be set.
	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&cudaResource, glBufferObj, cudaGraphicsMapFlagsNone));

	// In addition to a handle, the address of the original buffer in the device (GPU) memory will be needed
	// Mapping the shared resource and then requiesting a pointer to the mapped resource
	// devicePtr can be used by OpenGL as a pixel source directly by using the data in that memory
	 uchar4* devicePtr;
	 size_t size;
	 HANDLE_ERROR(cudaGraphicsMapResources(1, &cudaResource, NULL));
	 HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devicePtr, &size, cudaResource));

	// Launching the kernel
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);
	kernel_Interop << <grids, threads >> > (devicePtr);

	HANDLE_ERROR(cudaGraphicsUnmapResources(1, &cudaResource, NULL));

	// Setting up GLUT to run the main loop
	glutKeyboardFunc(key_func);
	glutDisplayFunc(draw_func);
	glutMainLoop();

	return 0;
}

static void draw_func(void)
{
	glDrawPixels(DIM, DIM, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glutSwapBuffers();
}

// Use ESC key to exit
static void key_func(unsigned char key, int x, int y)
{
	switch (key)
	{
		case 27:
			// clean up OpenGL and CUDA
			HANDLE_ERROR(cudaGraphicsUnregisterResource(cudaResource));
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
			glDeleteBuffers(1, &glBufferObj);
			exit(0);
	}
}


