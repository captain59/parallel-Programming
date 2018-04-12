#include "headers.h"

// Defining error limit for verification
#define errorLimit 1e-5

// defining function
void host_process_kernel_1(const float*, const float*, float*, int);
void host_process_kernel_2(const float*, float*, int);
void host_process_kernel_3(const float*, float*, int);

int main(void)
{
	// Error Code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;
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
	err = cudaMalloc((void **)&d_A, size);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to array A. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_B = NULL;
	err = cudaMalloc((void **)&d_B, size);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to array B. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_outKernel_1 = NULL;
	err = cudaMalloc((void **)&d_outKernel_1, size);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to array d_outKernel_1. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// copy data from host input arrays to device memory
	printf("Copy data from host arrays to device arrays\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to copy array A from host to device with error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to copy array B from host to device with error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Launch kernel 1
	// Dimensions for process_kernel1
	dim3 blocksPerGrid_1(4, 2, 2);
	dim3 threadsPerBlock_1(32, 32, 1);
	printf("CUDA kernel_1 launched with %d blocks of %d threads\n", blocksPerGrid_1.x*blocksPerGrid_1.y*blocksPerGrid_1.z, threadsPerBlock_1.x*threadsPerBlock_1.y*threadsPerBlock_1.z);
	process_kernel1<<< blocksPerGrid_1, threadsPerBlock_1>>>(d_A, d_B, d_outKernel_1, numElements);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		fprintf(stderr, " Failed to launch process_kernel_1 with error code %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// copy the device result from device memory to host memory
	printf("Copy data output of kernel1 from device memory to host memory\n");
	err = cudaMemcpy(h_outKernel_1, d_outKernel_1, size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to copy from device memory in kernel1 to host memory with error code %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Launch Kernel 2
	// Allocate host memory for result of process_kernel2
	float *h_outKernel_2 = (float*)malloc(size);
	// Allocate device memory for output of process_kernel2
	float *d_outKernel_2 = NULL;
	err = cudaMalloc((void **)&d_outKernel_2, size);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to array d_outKernel_2. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Dimensions for process_kernel2
	dim3 blocksPerGrid_2(2, 16,1);
	dim3 threadsPerBlock_2(8, 8, 8);
	printf("CUDA kernel_2 launched with %d blocks of %d threads\n", blocksPerGrid_2.x*blocksPerGrid_2.y*blocksPerGrid_2.z, threadsPerBlock_2.x*threadsPerBlock_2.y*threadsPerBlock_2.z);
	process_kernel2<<< blocksPerGrid_2, threadsPerBlock_2>>>(d_outKernel_1, d_outKernel_2,numElements);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		fprintf(stderr, " Failed to launch process_kernel_2 with error code %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("Copy data output of kernel2 from device memory to host memory\n");
	err = cudaMemcpy(h_outKernel_2, d_outKernel_2, size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		fprintf(stderr, " Failed to copy from device memory in kernel2 to host memory with error code %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Launching Kernel 3
	// Allocate host memory for result of process_kernel3
	float *h_outKernel_3 = (float*)malloc(size);
	// Allocate device memory for output of process_kernel3
	float *d_outKernel_3 = NULL;
	err = cudaMalloc((void **)&d_outKernel_3, size);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to array d_outKernel_3. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Dimensions for process_kernel3
	dim3 blocksPerGrid_3(32, 1, 1);
	dim3 threadsPerBlock_3(128, 4, 1);
	printf("CUDA kernel_3 launched with %d blocks of %d threads\n", blocksPerGrid_3.x*blocksPerGrid_3.y*blocksPerGrid_3.z, threadsPerBlock_3.x*threadsPerBlock_3.y*threadsPerBlock_3.z);
	process_kernel3<<< blocksPerGrid_3, threadsPerBlock_3 >>>(d_outKernel_2, d_outKernel_3, numElements);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		fprintf(stderr, " Failed to launch process_kernel_3 with error code %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("Copy data output of kernel3 from device memory to host memory\n");
	err = cudaMemcpy(h_outKernel_3, d_outKernel_3, size, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		fprintf(stderr, " Failed to copy from device memory in kernel3 to host memory with error code %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Verification Of Results
	// Output of kernel_1
	// Array for storing output of CPU computation of process_kernel1
	float *host_output_1 = (float*)malloc(size); 
	// Function to caculate the output of process_kernel1 using CPU
	host_process_kernel_1(h_A, h_B, host_output_1, numElements);
	for(int i=0; i < numElements; i++){
		if(fabs(host_output_1[i] - h_outKernel_1[i]) > errorLimit){
			fprintf(stderr, "Result Verification failed for Kernel_1 at element %d\n", i);
			exit(EXIT_FAILURE);
		}
	}
	printf("Test Passed for results of Kernel_1\n");
	//Verification of Results
	// Output of Kernel 2
	float *host_output_2 = (float*)malloc(size);
	host_process_kernel_2(host_output_1, host_output_2, numElements);
	for(int i=0; i < numElements; i++){
		if(fabs(host_output_2[i] - h_outKernel_2[i]) > errorLimit){
			fprintf(stderr, "Result Verification failed for Kernel_2 at element %d\n", i);
			exit(EXIT_FAILURE);
		}
	}
	printf("Test Passed for results of Kernel_2\n");
	//Verification of Results
	// Output of Kernel 3
	float *host_output_3 = (float*)malloc(size);
	host_process_kernel_3(host_output_2, host_output_3, numElements);
	for(int i=0; i < numElements; i++){
		if(fabs(host_output_3[i] - h_outKernel_3[i]) > errorLimit){
			fprintf(stderr, "Result Verification failed for Kernel_3 at element %d\n", i);
			exit(EXIT_FAILURE);
		}
	}
	printf("Test Passed for results of Kernel_3\n");
	printf("All Test Cases Passed\n");
	// Free device global memory
	err = cudaFree(d_A);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to free device array d_A with error code %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	err = cudaFree(d_B);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to free device array d_B with error code %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	err = cudaFree(d_outKernel_1);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to free device array d_outKernel_1 with error code %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	err = cudaFree(d_outKernel_2);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to free device array d_outKernel_2 with error code %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	err = cudaFree(d_outKernel_3);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to free device array d_outKernel_3 with error code %s \n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	printf("All Device Memory Freed\n");
	// Free Host memory
	free(h_A);
	free(h_B);
	free(h_outKernel_1);
	free(h_outKernel_2);
	free(h_outKernel_3);
	free(host_output_1);
	free(host_output_2);
	free(host_output_3);
	printf("All Host Memory Freed\n");
	// Reset the Device and Exit
	err = cudaDeviceReset();
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	printf("Done\n");
	return 0;
}

/*
Function to replicate process_kernel1
*/
void host_process_kernel_1(const float *A, const float *B, float *C, int numElements){
	for(int i = 0; i < numElements; i++){
		C[i] = (float)sin(A[i]) + (float)cos(B[i]);
	}
}
/*
Function to relpicate process_kernel2
*/
void host_process_kernel_2(const float *input, float *output, int numElements){
	for(int i = 0; i < numElements; i++){
		output[i] = (float)log(fabs(input[i]));
	}
}
/*
Function to replicate process_kernel3
*/
void host_process_kernel_3(const float *input, float *output, int numElements){
	for(int i = 0; i < numElements; i++){
		output[i] = (float)sqrt(input[i]);
	}
}