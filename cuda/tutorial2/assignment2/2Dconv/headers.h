#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void convolution1D(const int*, const int*, int*, int, int);

__global__ void convolution1D(const float*, const float*, float*, int, int);
