#include <cuda.h>

__global__ pictureKernel(float* d_Pin, float* d_Pout, int n, int m)
{
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if (Row < m && Col < n) {
    d_Pout[Row * n + Col] = 2.0 * d_Pin[Row * n + Col];
  }
}