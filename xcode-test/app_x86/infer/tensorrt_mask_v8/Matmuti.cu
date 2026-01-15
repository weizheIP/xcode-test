#include "muti.cuh"
#include <stdio.h>

__global__ void Muti(unsigned char *a, float *b,int size,int depart)
{
    int f = blockIdx.x*blockDim.x + threadIdx.x;

    if (f >= size) return;

    b[f+depart] = (float)(a[f])/255.;


}
// 3x32 的thread , [3,32] x [32,1] ,3 result
// 对应位置相乘,result 为队列后续做相加
__global__ void MatMut(float* dim1,float* dim2,float* result)
{
    int block_col = blockIdx.x;
    int block_row = blockIdx.y;
    int row = threadIdx.y;
    int col = threadIdx.x;
    // printf("%d\n",row);
    // int col = blockDim.x * blockIdx.x + threadIdx.x; 
    printf("%d:%d:%d:%d\n",block_row,block_col,row,col);
    // if (f >= size) return;
    // b[f+depart] = (float)(a[f])/255.;
    result[block_col] += (dim1[block_col*col] * dim2[col]);
}
// 所有相加 , block = 1,thread = size
__global__ void AddAll(float *src,int size,float result)
{
    int block_col = blockIdx.x;
    int block_row = blockIdx.y;
    int row = threadIdx.y;
    int col = threadIdx.x;
    // printf("%d\n",row);
    // int col = blockDim.x * blockIdx.x + threadIdx.x; 
    printf("%d:%d:%d:%d\n",block_row,block_col,row,col);
    // if (f >= size) return;
    resulta += src[col];
    // b[f+depart] = (float)(a[f])/255.;


}

void MutiFun(unsigned char *a, float *b,int size,int depart)
{
    dim3 dimBlock = (20);
    dim3 dimGrid = ((size + 20 - 1) / 20);
    Muti << <dimGrid, dimBlock >> > (a , b , size,depart);
    cudaDeviceSynchronize();
}
int getThreadNum()
{
    cudaDeviceProp prop;
    int count;
 
    (cudaGetDeviceCount(&count));
    printf("gpu num %d\n", count);
    (cudaGetDeviceProperties(&prop, 0));
    printf("max thread num: %d\n", prop.maxThreadsPerBlock);
    printf("max grid dimensions: %d, %d, %d)\n",
     prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    return prop.maxThreadsPerBlock;
}

int main()
{getThreadNum();
// return 0;
    int list[3][32];
    int list32[32] = {1};
    list[0] = list[1] = list[2] = list32;
    int list32_2[32] = {2};
    int list_4[4][32];
    list_4[0] = list_4[1] = list_4[2] = list_4[3] = list32_2;
    
    void* _list,_list_4,_;
    cudaMalloc(&ll,sizeof(list));
    cudaMemcpy(ll,&(list[0]), sizeof(list), cudaMemcpyHostToDevice);
    cudaMemcpy(ll,&(list[0]), sizeof(list), cudaMemcpyHostToDevice);
    cudaMemcpy(ll,&(list[0]), sizeof(list), cudaMemcpyHostToDevice);
    // SIZE = 4

    dim3 dimBlock (32,32);
    dim3 dimGrid(3);//((10 + 20 - 1) / 20);
    Test << <dimGrid, dimBlock >> > ((int*)ll);
    cudaDeviceSynchronize();
}






