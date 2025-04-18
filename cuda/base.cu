#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

static inline int divup(int a, int b) {
    return (a + b - 1) / b;
}

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}


__global__ void mykernel(float* r, const float* d, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= n || j >= n) return;
    float v = HUGE_VALF;
    for (int k = 0; k < n; ++k) {
        float x = d[n * i + k];
        float y = d[n * k + j];
        float z = x + y;
        v = min(v, z);
    }
    r[n * i + j] = v;
}


void step(float* r, const float* d, int n) {
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, n * n * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, n * n * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, d, n * n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(n, dimBlock.x), divup(n, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(rGPU, dGPU, n);
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(r, rGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
}
