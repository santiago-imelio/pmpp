#include <cuda.h>
#include <iostream>

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
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if (Row < width && Col < width) {
    float P_value = 0;

    for (int k = 0; k < width; k++) {
      P_value += d_M[Row * width + k] * d_N[k * width + Col];
    }

    d_P[Row * width + Col] = P_value;
  }
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

  dim3 dimBlock(1);
  dim3 dimGrid(16, 16);
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