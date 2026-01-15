// #pragma once

// // #include "cuda_runtime.h"
// // #include "device_launch_parameters.h"

// extern "C" void MutiFun2(unsigned char *a, float *b,int size,int depart);
// extern "C" void MatMuti(float* dim1,float *dim2,float* result,int dim1size[2],int dim2size[2], int dx);


#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void MutiFun2(unsigned char *a, float *b,int size,int depart);
void MatMuti(float* dim1,float *dim2,float* result,int dim1size[2],int dim2size[2],int dx1);

#ifdef __cplusplus
}
#endif