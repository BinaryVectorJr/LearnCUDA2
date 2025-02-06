// LearnCUDA2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

//#include "./GPU_VectorAdd.cuh"
//#include "./GPU_Atomics.cuh"
#include "./GPU_Interop.cuh"

int main()
{
    //call_VectorAdd();
    //call_AtomicsOnGPU();

    // Setting default values for GLUT
    // int _argc = 1;
    // char* _argv[1] = { (char*)"Default" };
    // call_Interop(_argc,_argv);

    Interop_Anim();

}