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
void pictureKernel(float* d_Pin, float* d_Pout, int n, int m, float factor)
{
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int Col = blockIdx.x * blockDim.x + threadIdx.x;

  if (Row < m && Col < n) {
    d_Pout[Row * n + Col] = factor * d_Pin[Row * n + Col];
  }
}

void rescale(float* Pin, float* Pout, int n, int m, float factor)
{
  int picSize = n * m * sizeof(float);

  float *d_Pin, *d_Pout;

  gpuErrchk(cudaMalloc((void **) &d_Pin, picSize));
  gpuErrchk(cudaMemcpy(d_Pin, Pin, picSize, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc((void **) &d_Pout, picSize));

  dim3 dimBlock(ceil(n/16.0), ceil(m/16.0), 1);
  dim3 dimGrid(16, 16, 1);
  pictureKernel<<<dimGrid, dimBlock>>>(d_Pin, d_Pout, n, m, factor);

  gpuErrchk(cudaMemcpy(Pout, d_Pout, picSize, cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(d_Pin));
  gpuErrchk(cudaFree(d_Pout));
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
  int n = 6;
  int m = 6;

  float* mat = iota(m, n);
  float* result = new float[m * n];

  rescale(mat, result, n, m, 2.0);

  for (int k = 0; k < m * n; k++) {
    if ((k + 1) % m == 0) {
      cout << result[k] << endl;
    } else {
      cout << result[k] << " ";
    }
  }

  delete mat;
  delete result;
}