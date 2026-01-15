#include "muti2.cuh"
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void Muti2(unsigned char *a, float *b,int size,int depart)
{
    int f = blockIdx.x*blockDim.x + threadIdx.x;

    if (f >= size) return;
    float i_d = (float)(a[f]);
    b[f+depart] = (i_d)/255.0f;


}


void MutiFun2(unsigned char *a, float *b,int size,int depart)
{
    dim3 dimBlock (640);
    dim3 dimGrid  ((size) / 640);
    Muti2 << <dimGrid, dimBlock >> > (a , b , size,depart);
    cudaDeviceSynchronize();
}

// int getThreadNum()
// {
//     cudaDeviceProp prop;
//     int count;
 
//     (cudaGetDeviceCount(&count));
//     printf("gpu num %d\n", count);
//     (cudaGetDeviceProperties(&prop, 0));
//     printf("max thread num: %d\n", prop.maxThreadsPerBlock);
//     printf("max grid dimensions: %d, %d, %d)\n",
//      prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
//     return prop.maxThreadsPerBlock;
// }







// 3x32 的thread , [3,32] x [32,N] ,[3,N] result
// 对应位置相乘,result 为队列后续做相加
// 
__global__ void MatMut_CUDA(float* dim1,float* dim2,float* result,int* dim1s,int* dim2s)
{
    // int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    // int row = threadIdx.y;
    int col = threadIdx.x;
    // printf("result[block_col*dim1s[0]+col]:%.2f",result[block_col*dim1s[0]+col]);
    // printf("index:%d\n",index[0]);
    for(int dim_index_t = 0;dim_index_t<dim1s[1];dim_index_t++)
    {
        result[block_col*dim1s[0]+col] += (dim1[(col*dim1s[1])+dim_index_t] * dim2[block_col*dim1s[1]+dim_index_t]);
        // printf("index:%d:%d:%d:%f,data:%f*%f,%f\n",(block_col*dim1s[1])+dim_index_t,dim_index_t,(dim1[(block_col*dim1s[1])+dim_index_t] * dim2[dim_index_t]),result[block_col],
        // dim1[(block_col+1)*dim_index_t],dim2[dim_index_t],result[block_col]);
    }
}

// dim2 应该是转置后的矩阵，n x m -> m x n,方便用于计算
// result [dim1size[0],dim2size[1]]
void MatMuti(float* dim1,float *dim2,float* result,int dim1size[2],int dim2size[2],int dx1)
{
    int dx = 640;//1280->320x320 800->200x200;// 102400 的整除数
    dim3 dimBlock (dim1size[0]);
    dim3 dimGrid(dx);//((10 + 20 - 1) / 20);
    // void* dim1s,*dim2s,*index;
    void* dim1s,*dim2s;
    cudaMalloc(&dim1s,2*sizeof(int));
    cudaMalloc(&dim2s,2*sizeof(int));
    // cudaMalloc(&index,1*sizeof(int));
    cudaMemcpy(dim1s,dim1size, 2*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dim2s,dim2size, 2*sizeof(int), cudaMemcpyHostToDevice);
    for(int i = 0;i<dim2size[1];i+=dx){
        // cudaMemcpy(index,&i, 1*sizeof(int), cudaMemcpyHostToDevice);b 
        MatMut_CUDA << <dimGrid, dimBlock >> > (dim1,&(dim2[i*dim1size[1]]),&(result[i*dim1size[0]]),(int*)dim1s,(int*)dim2s);
        // MatMut_CUDA << <dimGrid, dimBlock >> > (dim1,(dim2),(result),(int*)dim1s,(int*)dim2s,(int*)index);
    }
    // MatMut << <dimGrid, dimBlock >> > (dim1,dim2,result);
    cudaFree(dim1s);
    cudaFree(dim2s);
    // cudaFree(index);
    cudaDeviceSynchronize();
}

