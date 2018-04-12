#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void process_kernel1(const float* input1, const float* input2, float* output, int numElements){

	int blockNum = blockIdx.z*(gridDim.x*gridDim.y) + blockIdx.y*gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int globalThreadId = blockNum*(blockDim.x*blockDim.y*blockDim.z) + threadNum;

	if(globalThreadId < numElements)
		output[globalThreadId] = (float)sin(input1[globalThreadId]) + (float)cos(input2[globalThreadId]);

}

// Device Function for process_kernel2

__global__ void process_kernel2(const float* input, float* output, int numElements){

	int blockNum = blockIdx.z*(gridDim.x*gridDim.y) + blockIdx.y*gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int globalThreadId = blockNum*(blockDim.x*blockDim.y*blockDim.z) + threadNum;

	if(globalThreadId < numElements)
		output[globalThreadId] = (float)log(fabs(input[globalThreadId]));
}

// Device Function for process_kernel3

__global__ void process_kernel3(const float* input, float* output, int numElements){

	int blockNum = blockIdx.z*(gridDim.x*gridDim.y) + blockIdx.y*gridDim.x + blockIdx.x;
	int threadNum = threadIdx.z*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
	int globalThreadId = blockNum*(blockDim.x*blockDim.y*blockDim.z) + threadNum;

	if(globalThreadId < numElements)
		output[globalThreadId] = (float)sqrt(input[globalThreadId]);
}


int main(void)
{
	// Size of Array elements
	int numElements = 32*32*4*2*2;
	size_t size = numElements*sizeof(float);
	printf("[Operations on Kernel 1 on %d elements] \n", numElements);
	// Allocate the host input arrays 
	float *h_A = (float*)malloc(size);
	float *h_B = (float*)malloc(size);
	// Allocate the host array for result
	float *h_outKernel_1 = (float*)malloc(size);
	// Verify that the allocations are successful
	if(h_A == NULL || h_B == NULL || h_outKernel_1 == NULL){
		fprintf(stderr, "Failed to allocate size for Host arrays\n");
		exit(EXIT_FAILURE);
	}
	// Initialize the host vectors with random values
	for(int i=0; i < numElements; i++){
		h_A[i] = rand()/(float)RAND_MAX;
		h_B[i] = rand()/(float)RAND_MAX;
	}
	// Allocate the device input arrays
	float *d_A = NULL;
	cudaMalloc((void **)&d_A, size);
	
	float *d_B = NULL;
	cudaMalloc((void **)&d_B, size);
	
	float *d_outKernel_1 = NULL;
	cudaMalloc((void **)&d_outKernel_1, size);
	
	// copy data from host input arrays to device memory
	printf("Copy data from host arrays to device arrays\n");
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	
	dim3 blocksPerGrid_1(4, 2, 2);
	dim3 threadsPerBlock_1(32, 32, 1);
	process_kernel1<<< blocksPerGrid_1, threadsPerBlock_1>>>(d_A, d_B, d_outKernel_1, numElements);
	
	// copy the device result from device memory to host memory
	printf("Copy data output of kernel1 from device memory to host memory\n");
	cudaMemcpy(h_outKernel_1, d_outKernel_1, size, cudaMemcpyDeviceToHost);
	
	// Launch Kernel 2
	// Allocate host memory for result of process_kernel2
	float *h_outKernel_2 = (float*)malloc(size);
	// Allocate device memory for output of process_kernel2
	float *d_outKernel_2 = NULL;
	cudaMalloc((void **)&d_outKernel_2, size);
	
	// Dimensions for process_kernel2
	dim3 blocksPerGrid_2(2, 8, 1);
	dim3 threadsPerBlock_2(8, 8, 16);

	process_kernel2<<< blocksPerGrid_2, threadsPerBlock_2>>>(d_outKernel_1, d_outKernel_2,numElements);
	
	printf("Copy data output of kernel2 from device memory to host memory\n");
	cudaMemcpy(h_outKernel_2, d_outKernel_2, size, cudaMemcpyDeviceToHost);
	
	// Launching Kernel 3
	// Allocate host memory for result of process_kernel3
	float *h_outKernel_3 = (float*)malloc(size);
	// Allocate device memory for output of process_kernel3
	float *d_outKernel_3 = NULL;
	cudaMalloc((void **)&d_outKernel_3, size);
	
	// Dimensions for process_kernel3
	dim3 blocksPerGrid_3(32, 1, 1);
	dim3 threadsPerBlock_3(128, 4, 1);
	process_kernel3<<< blocksPerGrid_3, threadsPerBlock_3 >>>(d_outKernel_2, d_outKernel_3, numElements);
	
	printf("Copy data output of kernel3 from device memory to host memory\n");
	cudaMemcpy(h_outKernel_3, d_outKernel_3, size, cudaMemcpyDeviceToHost);
	
	// Free device global memory
	cudaFree(d_A);
	
	cudaFree(d_B);
	
	cudaFree(d_outKernel_1);
	
	cudaFree(d_outKernel_2);
	
	cudaFree(d_outKernel_3);
	
	printf("All Device Memory Freed\n");
	// Free Host memory
	free(h_A);
	free(h_B);
	free(h_outKernel_1);
	free(h_outKernel_2);
	free(h_outKernel_3);
	printf("All Host Memory Freed\n");
	// Reset the Device and Exit
	cudaDeviceReset();
	
	printf("Done\n");
	return 0;
}