#include "headers.h"

__global__ void ReduceRowMajor(int *g_idata, int *g_odata, int size) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;	
	unsigned int tid = threadIdx.x;
	extern __shared__ int sdata[];
	sdata[tid] = 0;
	if(i < size)
		sdata[tid] = g_idata[i];
	__syncthreads();
	for(unsigned int s = 4; s < blockDim.x; s*=2) {
		if(tid%(2*s) == 0) {
			sdata[tid] += sdata[tid+s];
			sdata[tid+1] += sdata[tid+s+1];
			sdata[tid+2] += sdata[tid+s+2];
			sdata[tid+3] += sdata[tid+s+3];
		}	
		__syncthreads();
	}
	if(tid == 0) {
		g_odata[blockIdx.x*4] = sdata[0];
		g_odata[blockIdx.x*4+1] = sdata[1];
		g_odata[blockIdx.x*4+2] = sdata[2];
		g_odata[blockIdx.x*4+3] = sdata[3];
	}
}

__global__ void ReduceRowMajor2(int *g_idata, int *g_odata, int size) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;	
	unsigned int tid = threadIdx.x;
	extern __shared__ int sdata[];
	sdata[tid] = 0;
	if(i < size)
		sdata[tid] = g_idata[i];
	__syncthreads();
	for(unsigned int s = 4; s < blockDim.x; s*=2) {
		int index = 2*s*tid;
		if(index < blockDim.x) {
			sdata[index] += sdata[index+s];
			sdata[index+1] += sdata[index+s+1];
			sdata[index+2] += sdata[index+s+2];
			sdata[index+3] += sdata[index+s+3];
		}	
		__syncthreads();
	}
	if(tid == 0) {
		g_odata[blockIdx.x*4] = sdata[0];
		g_odata[blockIdx.x*4+1] = sdata[1];
		g_odata[blockIdx.x*4+2] = sdata[2];
		g_odata[blockIdx.x*4+3] = sdata[3];
	}
}

__global__ void ReduceRowMajor3(int *g_idata, int *g_odata, int size) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;	
	unsigned int tid = threadIdx.x;
	extern __shared__ int sdata[];
	sdata[tid] = 0;
	if(i < size)
		sdata[tid] = g_idata[i];
	__syncthreads();
	for(unsigned int s = blockDim.x/2; s > 3; s/=2) {
		if(tid < s) {
			sdata[tid] += sdata[tid+s];
		}	
		__syncthreads();
	}
	if(tid == 0) {
		g_odata[blockIdx.x*4] = sdata[0];
		g_odata[blockIdx.x*4+1] = sdata[1];
		g_odata[blockIdx.x*4+2] = sdata[2];
		g_odata[blockIdx.x*4+3] = sdata[3];
	}
}

__global__ void ReduceRowMajor5(int *g_idata, int *g_odata, int size) {
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;	
	unsigned int tid = threadIdx.x;
	extern __shared__ int sdata[];
	sdata[tid] = 0;
	if(i < size)
		sdata[tid] = g_idata[i];
	__syncthreads();
	for(unsigned int s = blockDim.x/2; s >= 32; s/=2) {
		if(tid < s) {
			sdata[tid] += sdata[tid+s];
		}	
		__syncthreads();
	}
	if(tid < 32) {
		warpReduce(sdata, tid, size);
	}
	if(tid == 0) {
		g_odata[blockIdx.x*4] = sdata[0];
		g_odata[blockIdx.x*4+1] = sdata[1];
		g_odata[blockIdx.x*4+2] = sdata[2];
		g_odata[blockIdx.x*4+3] = sdata[3];
	}
}

__device__ void warpReduce(volatile int* sdata, int tid, int n) {
	if(tid + 32 < n)
		sdata[tid] += sdata[tid+32];
	if(tid + 16 < n)
		sdata[tid] += sdata[tid+16];
	if(tid + 8 < n)	
		sdata[tid] += sdata[tid+8];
	if(tid + 4 < n)	
		sdata[tid] += sdata[tid+4];
}

