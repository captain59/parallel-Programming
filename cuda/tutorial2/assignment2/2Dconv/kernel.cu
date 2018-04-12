#include "headers.h"

__global__ void convolution1D(const int *d_arr, const int *d_conv, int *d_result, int N, int M) {
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int globalId = i*N + j;
	if(globalId < N) {
		int convSum = 0, cnum = 0, k = M/2;
		for(int i=-k; i<=k; i++) {
			if(globalId + i >= 0 && globalId + i < N && cnum < M) {
				convSum += d_arr[globalId + i]*d_conv[cnum];
			}
			cnum++;
		}
		d_result[globalId] = convSum;
	}
}

__global__ void convolution2D(const float *d_arr, const float *d_mask, float *d_result, int N) {
	int i = threadIdx.x/N;
	int j = threadIdx.x%N;
	int globalId = i*N + j;
	if(i < N && j< N) {
		float avgSum = 0;
		int id, cnum = 0;
		for(int p = i-1; p <= i+1; p++) {
			for(int q = j-1; q<= j+1; q++) {
				if(p >=0 && p < N && q>=0 && q < N) {
					id = p*N + q;
					avgSum += d_arr[id]*d_mask[cnum];
				}
				cnum++;
			}
		}
		d_result[globalId] = avgSum;
	}
}
