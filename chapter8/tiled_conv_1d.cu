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

#define MAX_MASK_WIDTH 5
#define TILE_WIDTH 4

__constant__ float M[MAX_MASK_WIDTH];

__global__ void tiled_conv_1d_basic_k(float* N, float* P, int Mask_Width, int Width)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float Nds[TILE_WIDTH + MAX_MASK_WIDTH - 1];

  int n = Mask_Width / 2;

  int halo_idx_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
  if (threadIdx.x >= blockDim.x - n) {
    Nds[threadIdx.x - (blockDim.x - n)] = (halo_idx_left < 0 ) ? 0 : N[halo_idx_left];
  }

  Nds[n + threadIdx.x] = N[blockIdx.x * blockDim.x + threadIdx.x];

  int halo_idx_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
  if (threadIdx.x < n) {
    Nds[n + blockIdx.x + threadIdx.x] = (halo_idx_right >= Width) ? 0 : N[halo_idx_right];
  }

  __syncthreads();

  float Pvalue = 0;
  for (int j = 0; j < Mask_Width; j++) {
    Pvalue += M[j] * Nds[threadIdx.x + j];
  }

  P[i] = Pvalue;
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

int main() {
  int width = 16;

  float* N = iota(1, width);
  float *d_P, *d_N;

  float *P = new float[width];
  float h_M[MAX_MASK_WIDTH];

  h_M[0] = 3;
  h_M[1] = 7;
  h_M[2] = 9;
  h_M[3] = 5;
  h_M[4] = 1;

  gpuErrchk(cudaMalloc((void **)&d_N, width * sizeof(float)));
  gpuErrchk(cudaMalloc((void **)&d_P, width * sizeof(float)));
  gpuErrchk(cudaMemcpy(d_N, N, width * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyToSymbol(M, h_M, MAX_MASK_WIDTH * sizeof(float)));

  tiled_conv_1d_basic_k<<<4, 4>>>(d_N, d_P, MAX_MASK_WIDTH, width);

  gpuErrchk(cudaMemcpy(P, d_P, width * sizeof(float), cudaMemcpyDeviceToHost));

  for (int k = 0; k < width; k++) {
    if ((k + 1) % width == 0) {
      cout << P[k] << endl;
    } else {
      cout << P[k] << " ";
    }
  }

  gpuErrchk(cudaFree(d_N));
  gpuErrchk(cudaFree(d_P));

  delete P;
  delete N;
}