# Summary up to Chapter V

## Motivation
Improvements in CPUs started to decelerate from 2003 onwards. The increases in clock frequency to improve perfomance no longer justified the power consumption and overheating that it generated.

The intrinsic parallelism of the GPU architecture that differentiated them from the conventional CPU allowed to continue application performance improvements.

![](cpu_vs_gpu.png)
*CPUs and GPUs have fundamentally different architectures.*

CPU architectures are _latency-oriented_. They are designed to minimize the execution time of a single thread.

On the other hand, GPU architectures are _throughput-oriented_, meaning that their design strives to maximize the total execution throughput of a massively large number of threads while allowing individual threads to take a potetially much longer time to execute.

Given the difference on their designs, CPUs and GPUs perform differently depending on the use case. Generally, CPUs will not perform well on tasks on which GPUs perform well, and vice versa. However, there could be cases where **an application might benefit from both sequential and parallel execution of code**.

## Introduction to CUDA
CUDA (_Compute Unified Device Architecture_) is a programming model introduced by NVIDIA in 2007 that is designed to support joint CPU-GPU execution of code. It leverages the parallel compute engine in NVIDIA GPUs to solve complex computational problems in a more efficient way than on a CPU.

CUDA empowers **heterogeneous computing**. In a single application, a developer can write both sequential code that is executed in a _host_ (CPU) and parallel code that is executed in a _device_ (NVIDIA GPU).

### CUDA C/C++
CUDA C is a superset of the C language. CUDA C also supports a growing set of C++, which is why it's also referred as CUDA C/C++.

CUDA programs are compiled using _NVIDIA C Compiler_ (NVCC).

### CUDA Exploits Data Parallelism
Many modern software applications have sections that exhibit a rich amount of **_data parallelism_**, a phenomenon that allows arithmetic operations to be safely performed on different parts of a data structure in parallel. CUDA devices accelerate the execution of these applications by utilizing their massive number of arithmetic units to these data-parallel sections.

![](../chapter3/data-parallelism.png)
*Vector addition exhibits data parallelism. Each sum $A[i] + C[i]$ can be safely performed in parallel.*

CUDA C source files are a mixture of host code ande device code. Host code is plain C/C++ code, compiled with the host's standard C/C++ compiler and run as traditional CPU process. **Device code** is marked with CUDA keywords `__global__` and `__device__` to differentiate data-parallel functions.

A **host** function is a traditional C function, optionally marked with `__host__` label. Host functions are compiled and run on a CPU process. Host functions cannot be called from device code.

A **_kernel_** is a function marked with `__global__` keyword. Kernels act as the entrypoint from host functions to perform computations in the GPU.

A **device function** is a function marked with `__device__` keyword, and can be called from kernels or other device functions, but not from host functions.

## CUDA Threads

### Thread organization
In CUDA, a kernel function specifies code to be executed by all threads during a parallel phase. When the host code launches a kernel, the CUDA runtime system generates a **grid** of threads that are organized in a two-level hierarchy. A grid consists of **blocks**, all of the same size.

![](../chapter4/thread-organization.png)
*Example: kernel 1 will be launched using a grid of 3 by 2 blocks, each of 5 by 3 threads.*

Grids and blocks can have up to 3 dimensions. Dimensions are specified in the *kernel execution configuration* using CUDA `dim3` structs.
```c
// kernel 1 execution configuration
dim3 dimBlock(5, 3);
dim3 dimGrid(3, 2);
kernel1<<<dimGrid, dimBlock>>>();
```

### Threads to data mapping
The configurable thread organization is essentially what enables CUDA to exploit data parallelism in order to boost performance. By mapping threads to different parts of the data—i.e. memory locations—, each part can be processed in parallel by the corresponding thread(s).

Kernel functions are executed by all threads in the specified grid. One can access the coordinates of the "current" thread using `threadIdx.{x,y,z}` as well as the "current" block using `blockIdx.{x,y,z}`. Each thread coordinate can be identified by the combination
```c
blockIdx.t * blockDim.t + threadIdx.t; // t in {x,y,z}
```
In essence, CUDA thread organization allows us to programatically map these thread identifiers to memory locations where the data that we want to process is stored, enabling threads to perform parallel data access.

Even though parallel data access brings significant performance boosts, some times memory bottlenecks arise due to poor management of the different memory types in the device, especially when dealing with a massively large number of threads.

The way to deal with these sort of issues is by making proper use of the CUDA device memories and ensure coalesced memory access.

## CUDA Memories

![](../chapter5/cuda-memory.png)

### Global memory
This memory is the largest and slowest memory type. It is implemented with DRAM technology. Using `cudaMalloc` and `cudaMemcpy` will make use global memory. It is critical to minimize global memory transactions since it has major implications on kernel performance.

### Constant memory
Constant variables are stored in the global memory, but are cached for efficient access. With appropiate access patterns, constant memory access is extremely fast and parallel.

### Registers
Registers are on-chip, fast access memories. These are much smaller than global memory, and each thread has its own set of register files that store their private data. When register files are exhausted, data is spilled into **local memory**, which is physically stored in the same location as global memory.

### Shared memory
Accessing shared variables from the shared memory is extremelly fast and highly parallel. Scope of a shared variable is within a thread block. A private version of the variable is created and used by each thread block during kernel execution.

![](../chapter5/cuda-variables.png)
*Keywords to specify the memory type for variables.*

## Advanced Topics
### Barrier Synchronization
A barrier for a group of threads means any thread must stop at this point in the code and cannot proceed until all other threads/processes reach this barrier. It provides a mean for coordination between threads of the same block.

In CUDA, the `__syncthreads()` statement acts as such barrier.
- When `__syncthreads()` is present, it must be executed by all threads in a block.
- When a `__syncthreads()` statement is placed in an `if` statement, either all threads in a block execute the path that includes `__syncthreads()` or none of them does.
- For an `if-else` statement, if each path has a `__syncthreads()` statement, either all threads execute the `__syncthreads()` on the `if` path or all of them execute the `else` path. **These are two different barrier synchronization points**.

It is a responsability of the programmers to write their code so that these requirements are satisfied.

### Thread Scheduling in Hardware

A **warp** is a set of 32 threads within a thread block such that all the threads in a warp execute the same instruction.

<img title="warps" alt="Alt text" src="../chapter4/warps.png">

The execution of warps are implemented by a SIMD (single instruction, multiple data) hardware.
- Each thread from a warp is executed in a processing unit, akin to a CPU core.
- All processing units are controlled by the same instruction, but these differ in the data operand values of the register files.

![](sm-design.png)
*SIMD processor design.*

In general, there are fewer processing units than the number of threads assigned to each SM. That is, an SM has only enough hardware to execute instructions from a small subset of all threads assigned to it, at any point in time.

Even though this may seem as a waste, it constitutes a **latency tolerance** mechanism; with enough warps around, the hardware will likely find a warp to execute at any point in time, thus making full use of execution hardware and preventing idle time. This is known as _zero-overhead thread scheduling_.

