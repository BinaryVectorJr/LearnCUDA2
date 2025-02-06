#define GL_GLEXT_PROTOTYPES
#pragma warning(disable : 4996)     // Needed because it was showing that cudaGlSetGLDevice was deprecated

#include "./CUDA_Helpers.cuh"
#include "./GL_Helper.cuh"

#define DIM 512

struct GPUAnimBitmap
{
    GLuint bufferObj;   // OpenGL buffer
    cudaGraphicsResource* resource;     // cuda resource that acts as a handle to the OpenGL resource
    int width, height;      // height and width of the image we will be generating
    void* dataBlock;        // contains the user data
    void (*fAnim)(uchar4*, void*, int);     //callback that gets calledto glutIdleFunc() and this is responsible for producing the image data rendered in the animation
    void (*animExit)(void*);        // callback that is called once the animation finishes so that cleanup code can be run
    void (*clickDrag)(void*, int, int, int, int);       // callback that implements click and drag mouse events and gets called after every click or every drag
    int dragStartX, dragStartY;

    GPUAnimBitmap(int w, int h, void* d)
    {
        width = w;
        height = h;
        dataBlock = d;
        clickDrag = NULL;

        // This part of the code is carried over from the original call code
        // Find a CUDA device and set it to graphics interop
        cudaDeviceProp prop;
        int dev;

        // memset(void* ptr, int x, size_t n) is used to fill a block of memory with a particular value
        // &prop = starting address of memory to be filled
        // x = value to be filled
        // n = number of bytes to be filled starting from &prop address
        memset(&prop, 0, sizeof(cudaDeviceProp));

        // Set GPU compute capability (1.0 or better)
        prop.major = 1;
        prop.minor = 0;
        HANDLE_ERROR(cudaChooseDevice(&dev, &prop));

        // From main.cpp, call to initialize glutInit() otherwise it throws errors
        int _argc = 1;
        char* _argv[1] = { (char*)"Default" };
        glutInit(&_argc, _argv);

        // Back again from the main call from the call function of interop
        // Initializing glutWindow
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
        glutInitWindowSize(width, height);
        glutCreateWindow("bitmap");

        //anim_and_exit((void(*)(uchar4*, void*, int))generate_frame, NULL);

        // Need to initialize glew
        glewInit();

        // Tell CUDA to set the device we found to use for interops
        HANDLE_ERROR(cudaGLSetGLDevice(dev));

        // Bringing over more code from main call
        // IMP: Interop operations are highly dependent on shared data buffers, which are the key components to OpenGL rendering
        // Passing data b/w OpenGL and CUDA, first step is to create buffers for both APIs
        // Step 1: Generate buffer
        glGenBuffers(1, &bufferObj);
        // Step 2: Bind the handle to a pixel buffer
        // Basically cudaResource is a handle that is used to refer to the actual GL buffer by CUDA runtime
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObj);
        // Step 3: Request OpenGL driver to allocate a buffer so that we can use it; GL_DYNAMIC_DRAW_ARB tells OpenGL that it will be modified repeatedly by the program; NULL means we have no data to pass to the buffer
        glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4, NULL, GL_DYNAMIC_DRAW_ARB);

        // Sharing (copying) from the OpenGL buffer to the CUDA runtime buffer
        HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, bufferObj, cudaGraphicsMapFlagsNone));
    }

    ~GPUAnimBitmap()
    {
        freeResources();
    }

    void freeResources()
    {
        HANDLE_ERROR(cudaGraphicsUnregisterResource(resource));

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glDeleteBuffers(1, &bufferObj);
    }

    long image_size(void) const { return width * height * 4; }

    void click_drag(void (*f)(void*, int, int, int, int)) 
    {
        clickDrag = f;
    }

    // GLUT Idle function - for Idle callbacks
    static void idle_func(void)
    {
        // Step 1: Map the shared buffer and retrieve a GPU pointer for the buffer
        static int ticks = 1;
        GPUAnimBitmap* gpuBitmap = *(get_bitmap_ptr());
        uchar4* devicePtr;
        size_t size;
        HANDLE_ERROR(cudaGraphicsMapResources(1, &(gpuBitmap->resource), NULL));
        HANDLE_ERROR(cudaGraphicsResourceGetMappedPointer((void**)&devicePtr, &size, gpuBitmap->resource));

        // Step 2: call the fAnim function to launch CUDA C kernel to fill the buffer in device_ptr with image data
        gpuBitmap->fAnim(devicePtr, gpuBitmap->dataBlock, ticks++);

        // Step 3: unmap the GPU pointer to release buffer for use by the OpenGL driver in rendering
        // This call is triggered by glutPostRedisplay()
        HANDLE_ERROR(cudaGraphicsUnmapResources(1, &(gpuBitmap->resource), NULL));
        glutPostRedisplay();
    }

    void anim_and_exit(void (*f)(uchar4*,void*, int), void(*e)(void*))
    {
        GPUAnimBitmap** bitmap = get_bitmap_ptr();
        *bitmap = this;
        fAnim = f;
        animExit = e;

        glutKeyboardFunc(Key);
        glutDisplayFunc(Draw);
        if (clickDrag != NULL)
            glutMouseFunc(mouse_func);
        glutIdleFunc(idle_func);
        glutMainLoop();
    }

    // static method used for glut callbacks
    static GPUAnimBitmap** get_bitmap_ptr(void)
    {
        static GPUAnimBitmap* gBitmap;
        return &gBitmap;
    }

    // static method used for glut callbacks
    static void mouse_func(int button, int state, int mx, int my) 
    {
        if (button == GLUT_LEFT_BUTTON) {
            GPUAnimBitmap* bitmap = *(get_bitmap_ptr());
            if (state == GLUT_DOWN) {
                bitmap->dragStartX = mx;
                bitmap->dragStartY = my;
            }
            else if (state == GLUT_UP) {
                bitmap->clickDrag(bitmap->dataBlock,
                    bitmap->dragStartX,
                    bitmap->dragStartY,
                    mx, my);
            }
        }
    }

    // static method used for glut callbacks
    static void Key(unsigned char key, int x, int y) {
        switch (key) {
        case 27:
            GPUAnimBitmap * bitmap = *(get_bitmap_ptr());
            if (bitmap->animExit)
                bitmap->animExit(bitmap->dataBlock);
            bitmap->freeResources();
            exit(0);
        }
    }

    // static method used for glut callbacks
    static void Draw(void) {
        GPUAnimBitmap* bitmap = *(get_bitmap_ptr());
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawPixels(bitmap->width, bitmap->height, GL_RGBA,
            GL_UNSIGNED_BYTE, 0);
        glutSwapBuffers();
    }
};

// Kernel that actually does the GPU calculation (the one that is used with the triple angle brackets)
__global__ void kernel_Interop(uchar4* ptr);
__global__ void kernel_Interop_Anim(uchar4* ptr, int ticks);

// Main program doing the memory allocation, transfer, and freeing as well as calling the kernel
// Has the same args as the kernel call above
void main_Interop();

// Call program that should be put in the "main" of the solution CPP
int call_Interop(int argc, char** argv);

// Call animated version
void Interop_Anim();

// Function call that calls the kernel to generate the data
void generate_frame(uchar4* pixels, void*, int ticks);

// CPU Callback functions
static void draw_func(void);
static void key_func(unsigned char key, int x, int y);

