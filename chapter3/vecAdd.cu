#include <cuda.h>
#include <stdio.h>

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
void vecAddKernel(float* A, float* B, float* C, int n)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;

  if (i<n) C[i] = A[i] + B[i];
}


void vecAdd(float *A, float *B, float *C, int n)
{
  int size = n * sizeof(float);

  float *d_A, *d_B, *d_C;

  gpuErrchk(cudaMalloc((void **) &d_A, size));
  gpuErrchk(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc((void **) &d_B, size));
  gpuErrchk(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

  gpuErrchk(cudaMalloc((void **) &d_C, size));

  vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

  gpuErrchk(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

  gpuErrchk(cudaFree(d_A));
  gpuErrchk(cudaFree(d_B));
  gpuErrchk(cudaFree(d_C));
}

int main()
{
  int n = 5;

  float *A = (float*)malloc(n * sizeof(float));
  float *B = (float*)malloc(n * sizeof(float));
  float *C = (float*)malloc(n * sizeof(float));

  for (int i = 0; i < n; i++) {
    A[i] = i;
    B[i] = 2 * i;
  }

  vecAdd(A, B, C, n);

  for (int i = 0; i < n; i++) {
    printf("C[%d] = %.1f\n", i, C[i]);
  }
}