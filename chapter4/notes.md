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

## Handling multidimensional data
The choice of 1D, 2D, or 3D thread organizations are usually based on the nature of the data. For instance, grayscale images are 2D arrays of pixels, so it is often convenient to use a 2D grid of 2D blocks to process the pixels.

## Matrix linearization
When dealing with dynamically stored arrays, the C compiler does not know before hand the number of items that the array will store, and this is by design. Thus, the number of columns in a dynamically allocated 2D array is not known at compile time.

As a result, programmers need to explicitly **linearize** (flatten) a dynamically allocated matrix into an equivalent 1D array.

<img title="linearization" alt="Alt text" src="array-linearization.png">

Let $M$ be a $m \times n$ matrix, where $M_{j,i}$ represents the element at row $j$ and column $i$.

### Row-major layout access formula

$$

M_{j,i} = M[j * n + i]

$$

### Column-major access formula

$$

M_{j,i} = M[i * m + j]

$$

In reality, both static and dynamic arrays in C are linearized. Static multidimensional arrays gets to use the bracket syntax because the dimensional information is given at compile time, and under the hood the compiler linearizes it into 1D equivalent.