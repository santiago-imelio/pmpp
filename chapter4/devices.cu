#include <cuda.h>
#include <iostream>

using namespace std;

int main() {
  int dev_count;

  cudaGetDeviceCount(&dev_count);

  cout << "CUDA Devices: " << dev_count << endl;

  cudaDeviceProp dev_props;
  for (int i = 0; i < dev_count; i++) {
    cudaGetDeviceProperties(&dev_props, i);

    cout << "maxThreadsPerBlock: " << dev_props.maxThreadsPerBlock << endl;
    cout << "maxThreads x: " << dev_props.maxThreadsDim[0] << endl;
    cout << "maxThreads y: " << dev_props.maxThreadsDim[1] << endl;
    cout << "maxThreads z: " << dev_props.maxThreadsDim[2] << endl;
    cout << "multiProcessorCount: " << dev_props.multiProcessorCount << endl;
    cout << "clockRate: " << dev_props.clockRate << endl;
    cout << " " << endl;
  }
}