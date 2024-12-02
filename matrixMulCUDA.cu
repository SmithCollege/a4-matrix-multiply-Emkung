#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define TILE_WIDTH 10
__global__ void MatrixMulCUDA(float* M, float* N, float* P, int Width) {
     int Row = blockIdx.y * blockDim.y + threadIdx.y;
     int Col = blockIdx.x * blockDim.x + threadIdx.x;
     if ((Row < Width) && (Col < Width)) {
	   float Pvalue = 0;
	   for (int k = 0; k < Width; ++k){
     	       Pvalue += M[Row * Width + k] * N[k * Width + Col];
	       }
	   P[Row * Width + Col] = Pvalue;
     }	 
}

int main() {
    float *M, *N, *P;
    int Width = TILE_WIDTH*TILE_WIDTH;
    cudaMallocManaged(&M, sizeof(float) * Width*Width);
    cudaMallocManaged(&N, sizeof(float) * Width*Width);
    cudaMallocManaged(&P, sizeof(float) * Width*Width);

    //initialize inputs
    for (int i = 0; i < Width*Width; i++){
    	M[i] = 1;
	N[i] = 1;
    }

    int block_num = ceil(1.0*Width/TILE_WIDTH);
    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(block_num, block_num,1);
    clock_t a = clock();
    MatrixMulCUDA<<<dimGrid, dimBlock>>>(M, N, P, Width);
    cudaDeviceSynchronize();
    clock_t b = clock() - a;
    for (int i = 0; i < Width*Width; i++){
        printf("%f ", P[i]);
    } 
    printf("time: %f ", (float) b/CLOCKS_PER_SEC);
    printf("\n");

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    cudaFree(M);
    cudaFree(N);
    cudaFree(P);
    return 0;
}