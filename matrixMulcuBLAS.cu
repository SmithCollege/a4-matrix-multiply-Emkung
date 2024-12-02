#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <cublas_v2.h>


int main() {
    int Width = 100;
    float *M, *N, *P;
    cudaMallocManaged(&M, sizeof(float) * Width*Width);
    cudaMallocManaged(&N, sizeof(float) * Width*Width);
    cudaMallocManaged(&P, sizeof(float) * Width*Width);

    //initialize inputs
    for (int i = 0; i < Width*Width; i++){
        M[i] = 1;
        N[i] = 1;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    clock_t a = clock();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Width, Width, Width, &alpha, N, Width, M, Width, &beta, P, Width);
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
