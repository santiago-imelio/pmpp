#include <cuda.h>
#include <iostream>

#define BLOCK_WIDTH 2
#define TILE_WIDTH 2

using namespace std;

// CUDA API error handling macro
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void matMulKernel(float* d_M, float* d_N, float* d_P, int width)
{
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // identify row and column of the d_P element to calculate
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float Pvalue = 0;
  // loop over the d_M and d_N tiles to compute d_P element
  for (int m = 0; m < width / TILE_WIDTH; m++) {
    // thread loads in shared memory its corresponding values in M and N tiles
    Mds[ty][tx] = d_M[Row * width + m * TILE_WIDTH + tx];
    Nds[ty][tx] = d_N[(m * TILE_WIDTH + ty) * width + Col];

    // wait for all threads in block to load Mds and Nds accordingly
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }

    // wait for all threads in block to use the loaded values in Mds and Nds accordingly
    __syncthreads();
  }

  d_P[Row * width + Col] = Pvalue;
}

void matmul(float* M, float* N, float* P, int width)
{
  int size = width * width * sizeof(float);

  float *d_M, *d_N, *d_P;

  gpuErrchk(cudaMalloc((void **) &d_M, size));
  gpuErrchk(cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc((void **) &d_N, size));
  gpuErrchk(cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc((void **) &d_P, size));

  int numBlocks = width / BLOCK_WIDTH;

  if (width % BLOCK_WIDTH) numBlocks++;

  dim3 dimGrid(numBlocks, numBlocks);
  dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

  matMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);

  gpuErrchk(cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(d_M));
  gpuErrchk(cudaFree(d_N));
  gpuErrchk(cudaFree(d_P));
}

float* iota(int m, int n)
{
  int len = m * n;
  float* mat = new float[len];

  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      mat[i * n + j] = i * n + j;
    }
  }

  return mat;
}

int main()
{
  int width = 4;

  float* mat1 = iota(width, width);
  float* mat2 = iota(width, width);
  float* result = new float[width * width];

  matmul(mat1, mat2, result, width);

  for (int k = 0; k < width * width; k++) {
    if ((k + 1) % width == 0) {
      cout << result[k] << endl;
    } else {
      cout << result[k] << " ";
    }
  }

  delete mat1;
  delete mat2;
  delete result;
}