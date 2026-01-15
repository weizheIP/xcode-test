#include "muti.cuh"
#include <stdio.h>

__global__ void Muti(unsigned char *a, float *b,int size,int depart)
{
    int f = blockIdx.x*blockDim.x + threadIdx.x;

    if (f >= size) return;

    b[f+depart] = (float)(a[f])/255.;
    // b[f] = a[f];
    // printf("%d:%d",f,a[f]);

    // for (int c = 0; c < 3; c++)
	// {
	// 	for (int i = 0; i < row; i++)
	// 	{
	// 		for (int j = 0; j < col; j++)
	// 		{
	// 			float pix = dstimg.at<cv::Vec3b>(i,j)[c];
	// 			blob[c * row * col + i * col + j] = pix / 255.0;
	// 		}
	// 	}
	// }

}

void MutiFun(unsigned char *a, float *b,int size,int depart)
{
    dim3 dimBlock = (20);
    dim3 dimGrid = ((size + 20 - 1) / 20);
    Muti << <dimGrid, dimBlock >> > (a , b , size,depart);
    cudaDeviceSynchronize();
}