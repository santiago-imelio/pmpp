## How to run CUDA code files in Google Colab
First configure the runtime to use GPU, go to _Runtime_ > _Change runtime type_. Then select T4 GPU. You can check that CUDA compiler, NVCC, is installed by running a cell with this code:

```
!nvcc --version
```

To run the code files just import them into the notebook files, then compile and run.

```
!nvcc vecAdd.cu -o vecAdd -arch=sm_75 && ./vecAdd
```