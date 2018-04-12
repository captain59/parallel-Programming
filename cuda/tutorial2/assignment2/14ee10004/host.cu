#include "headers.h"

#define N 10 // Array Size
#define M 5 // convolution mask

#define blockSize 32

void DisplayMatrix(const char*, int*, int);

int main(int argc, char const *argv[])
{
	// Error Code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;
	int gridSize = (int)ceil((float)N/(blockSize*blockSize));
	size_t sizeArr = N*sizeof(int);
	size_t sizeConv = M*sizeof(int);
	// Host Matrix
	int *h_arr = (int*)malloc(sizeArr);
	int *h_out1D = (int*)malloc(sizeArr);
	// Host convolution mask
	int *h_conv = (int*)malloc(sizeConv);
	// random value
	for(int i = 0; i < N; i++) {
		h_arr[i] = rand()%10;
	}
	for(int i = 0; i < M; i++) {
		h_conv[i] = rand()%10;
	}
	DisplayMatrix("Original Array", h_arr, N);
	DisplayMatrix("Convolution Mask", h_conv, M);
	// Device Matrix
	int *d_arr = NULL;
	err = cudaMalloc((void **)&d_arr, sizeArr);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to device array. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(d_arr, h_arr, sizeArr, cudaMemcpyHostToDevice);
	int *d_conv = NULL;
	err = cudaMalloc((void **)&d_conv, sizeConv);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to device convolution array. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(d_conv, h_conv, sizeConv, cudaMemcpyHostToDevice);
	int *d_1DconvResult = NULL;
	err = cudaMalloc((void **)&d_1DconvResult, sizeArr);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to device convolution result array. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Dimensions
	dim3 blocksPerGrid(gridSize);
	dim3 threadsPerBlock(32, 32);
	// Launching 1D convolution kernel
	convolution1D<<< blocksPerGrid, threadsPerBlock >>>(d_arr, d_conv, d_1DconvResult, N, M);
	fprintf(stderr, "Kernel Executed\n");
	err = cudaGetLastError();
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to Launch 1D Convolution Kernel");
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(h_out1D, d_1DconvResult, sizeArr, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		fprintf(stderr, " Failed to copy from device memory to host memory with error code %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("1D convolution Kernel Executed\n");
	DisplayMatrix("Convolution Result", h_out1D, N);
	// Reset the Device and Exit
	err = cudaDeviceReset();
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	printf("Done\n");
	return 0;
}

void DisplayMatrix(const char *inp, int *arr, int num) {
	printf("%s\n", inp);
	for(int i = 0; i < num; i++) {
		printf("%d \t", arr[i]);
	}
	printf("\n\n");
}
