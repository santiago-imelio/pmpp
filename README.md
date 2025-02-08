# PMPP
This repository contains notes and code snippets from the book _Programming Massively Parallel Processors (2nd Edition)_ by Kirk & Hwu.

## Running CUDA Programs

### Google Colab
The easiest way to run the CUDA C programs in this repo is with [Google Colab](https://colab.research.google.com/), in case you don't have access to an NVIDIA GPU.

First configure the runtime to use GPU, go to _Runtime_ > _Change runtime type_. Then select T4 GPU. You can check that CUDA compiler, NVCC, is installed by running a cell with this code:

```
!nvcc --version
```

Clone the repository

```
!git clone https://github.com/santiago-imelio/pmpp.git
```

To run the code files just import them into the notebook files, then compile and run.

```
!nvcc -arch=sm_75 pmpp/chapter3/vecAdd.cu -o vecAdd && ./vecAdd
```