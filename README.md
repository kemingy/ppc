# Programming Parallel Computers

Code for the course "[Programming Parallel Computers](https://ppc.cs.aalto.fi/)"

- Chapter 2: case study of the [shortcut](./shortcut/)
- Exercises
  - [correlated pairs](./correlated_pairs/)

## Methods

- linear scan the matrix (row/column based)
- use block to allow the compiler do auto-vectorization
- use GCC aligned vector for explicit vectorization
- re-schedule the memory access order to utilize the register cache
- re-design the memory access pattern to utilize the register cache
- prefetch 20 iterations
- re-order the inter and outer loop to cache more

## SIMD

- Latency: time to perform an operation from start to finish.
- Throughput: how many operations are completed per time unit, in the long run.

So roughly, the parallel num is `latency x throughput`.

## OpenMP

- basic

```cpp
#pragma omp parallel for
for (int i = 0; i < N; i++) {
  // do something
}
```

is equivalent to

```cpp
#pragma omp parallel
{
  #pragma omp for
  for (int i = 0; i < N; i++) {
    // do something
  }
}
```

- pre & post processing (run in every thread)

```cpp
a();
#pragma omp parallel
{
  before();
  #pragma omp for
  for (int i = 0; i < N; i++) {
    // do something
  }
  after();
}
b();
```

- pre & post processing without waiting all threads in the loop

```cpp
a();
#pragma omp parallel
{
  before();
  #pragma omp for nowait
  for (int i = 0; i < N; i++) {
    // do something
  }
  after();
}
b();
```

- critical section that each thread will run one by one

```cpp
a();
#pragma omp parallel
{
  before();
  #pragma omp for
  for (int i = 0; i < N; i++) {
    // do something
  }
  #pragma omp critical
  {
    // do something one by one thread
  }
  after();
}
b();
```

- single section that only one thread will run

```cpp
a();
#pragma omp parallel
{
  before();
  #pragma omp for
  for (int i = 0; i < N; i++) {
    // do something
  }
  #pragma omp single
  {
    // do something only once by one thread
  }
  after();
}
b();
```

- static schedule

Normally, thread 0 will do 0-n jobs and thread 1 will do n-2n jobs, which is good for linear memory access. But if you want thread 0 to do 0,n,2n jobs and thread 1 to do 1,n+1,2n+1 jobs, you can use `schedule(static, 1)`. `1` is the chunk size.

```cpp
#pragma omp parallel for schedule(static, 1)
for (int i = 0; i < N; i++) {
  // do something
}
```

- dynamic schedule (requires communication between threads)

```cpp
#pragma omp parallel for schedule(dynamic, 1)
for (int i = 0; i < N; i++) {
  // do something
}
```

- nested

From the outer loop, if the outer loop is too short to parallelize

```cpp
#pragma omp parallel for
for (int i = 0; i < N; i++) {
  for (int j = 0; j < M; j++) {
    // do something
  }
}
```

We can nest it by using `collapse(2)`

```cpp
#pragma omp parallel for collapse(2)
for (int i = 0; i < N; i++) {
  for (int j = 0; j < M; j++) {
    // do something
  }
}
```

Or manually flat the two loops:

```cpp
#pragma omp parallel for
for (int ij = 0; ij < M * N; ij++) {
  int i = ij / M;
  int j = ij % M;
  // do something
}
```

- tasks

`task` tells the following block of code could be executed in another thread.

```cpp
before();
#pragma omp parallel
#pragma omp single
{
  foo(1);
  #pragma omp task
  foo(2);
  #pragma omp task
  foo(3);
  foo(4);
}
after();
```

## HyperThreading

> [!NOTE]
> HyperThreading does not help with the maximum throughput of the CPU, but it can help keep the CPU busy by providing two instruction streams for each CPU core.
> That is to say, it helps with parallelism in large scale (multi-threads), but 
> doesn't help with parallelism in small scale (vectorization, instruction-level parallelism).

## OpenMP memory model

OpenMP performs a flush automatically whenever you enter or leave a `parallel`/`critical` region.

From the perspective of the hardware, cache memory would like to keep track of full **cache lines**, which are 64-byte units. Modifying adjacent variables in memory may require communication between CPU cores to make sure no data is lost.

Otherwise, you can use `atomic` if it's the case:

```cpp
#pragma omp parallel for
for (int i = 0; i < N; i++) {
  // do something
  #pragma omp atomic
  x ++;
}
```

An example:

```cpp
int sum_shared = 0;
#pragma omp parallel
{
  // private for each thread
  int sum_local = 0;
  #pragma omp for
  for (int i = 0; i < N; i++) {
    sum_local += i;
  }
  #pragma omp atomic
  sum_shared += sum_local;
}
print(sum_shared);
```

## OpenMP functions

- outside the `parallel`
  - `omp_get_max_threads()`
- inside the `parallel`
  - `omp_get_thread_num()`: current thread identifier, starts from 0
  - `omp_get_num_threads()`: threads number using in this region

To control the number of threads in a region:

```cpp
#pragma omp parallel num_threads(4)
{
  // do something
}
```
