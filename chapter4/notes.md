# IV - Data-parallel Execution Model

## CUDA Thread Organization
All CUDA threads in a grid execute the same kernel function and they rely on coordinates to distinguish themselves from each other and to identify the appropiate portion of the data to process.

Threads are organized in a two-level hierarchy:  grids that consist in one or more blocks, and each block consisting in one or more threads.

Blocks and threads are identified by indexes, which can be accessed via the following built-in variables:
- `blockIdx.x` returns the `x` coordinate of the block index.
- `threadIdx.x` returns the `x` coordinate of the thread index.

<img title="thread org" alt="Alt text" src="thread-organization.png">

In general, a grid is a 3D array of blocks, and each block is a 3D array of threads. So in essence, either a block or a thread can be located by a $(x,y,z)$ tuple.

Just like the indexes, the grid dimensions and block dimensions can be accessed via `gridDim` and `blockDim` respectively.

## Treating multidimensional data
The choice of 1D, 2D, or 3D thread organizations are usually based on the nature of the data.