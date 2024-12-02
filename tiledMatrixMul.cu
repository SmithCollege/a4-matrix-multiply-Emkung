#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define TILE_WIDTH 2
__global__ void TiledMatrixMul(float* M, float* N, float* P, int Width) {
     __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
     __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
     int bx = blockIdx.x;
     int by = blockIdx.y;
     int tx = threadIdx.x;
     int ty = threadIdx.y;

     int Row = by * TILE_WIDTH + ty;
     int Col = bx * TILE_WIDTH + tx;
     float Pvalue = 0;
     for (int m = 0; m < Width/TILE_WIDTH; ++m){ //why not just TILE_WIDTH
     	 subTileM[ty][tx] = M[Row*Width + m*TILE_WIDTH + tx];
	 subTileN[ty][tx] = N[(m*TILE_WIDTH + ty) * Width + Col];
	 __syncthreads();
	 for (int k = 0; k <  TILE_WIDTH; ++k){
	     Pvalue += subTileM[ty][k] * subTileN[k][tx];
	 }
	 __syncthreads();
     }
     P[Row * Width + Col] = Pvalue;
}

int main() {
    float *M, *N, *P;
    int Width = 1000;
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
    TiledMatrixMul<<<dimGrid, dimBlock>>>(M, N, P, Width);
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