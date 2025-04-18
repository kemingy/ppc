# CUDA

## Basic

```c
__device__ void foo(int i, int j) {}

__global__ void mykernel() {
    int i = blockIdx.x;
    int j = threadIdx.x;
    foo(i, j);
}

int main() {
    mykernel<<<100, 128>>>();
    cudaDeviceSynchronize();
}
```

- `<<<100, 128>>>` means 100 blocks of threads, each block should have 128 threads.
- `__global__` indicates that is a **kernal**, which can access `blockIdx` and `threadIdx`
- `__device__` indicates that is supposed to be executed on the GPU
- `dim3` create two or three dimensional indexes (access by `x,y,z`)

> [!Note]
> GPU will divide each block in smaller units of work called "wraps". Each wraps
> consists of 32 threads. So it's resonable to pick of a round number for threads
> in each block. It's limited to 1024 threads per block.

