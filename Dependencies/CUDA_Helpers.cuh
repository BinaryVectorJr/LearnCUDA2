/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 * EDITED BY: Angshuman 'Moz' Mazumdar
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>
#include <iostream>
#include <stdio.h>

static void HandleError(cudaError_t err,const char* file,int line) 
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

template< typename T >
void swap(T& a, T& b) {
    T t = a;
    a = b;
    b = t;
}


void* big_random_block(int size);

int* big_random_block_int(int size);


// Common kernels
__device__ unsigned char value(float n1, float n2, int hue);

__global__ void float_to_color(unsigned char* optr, const float* outSrc);

__global__ void float_to_color(uchar4* optr, const float* outSrc);



