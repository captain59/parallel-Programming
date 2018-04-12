#include "headers.h"

#define N 10 // Array Size
#define M 5 // convolution mask
#define mask2D 9 // 2D mask size

#define blockSize 32

void DisplayMatrix1D(const char*, int*, int);
void DisplayMatrix2D(const char*, float*, int);
void verify1Dresult(int*, int*, int*);

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
	DisplayMatrix1D("Original Array", h_arr, N);
	DisplayMatrix1D("Convolution Mask", h_conv, M);
	// Device Matrix
	int *d_arr = NULL;
	err = cudaMalloc((void **)&d_arr, sizeArr);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to device array. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// copying from Host to Device
	cudaMemcpy(d_arr, h_arr, sizeArr, cudaMemcpyHostToDevice);
	int *d_conv = NULL;
	err = cudaMalloc((void **)&d_conv, sizeConv);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to device convolution array. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// copying from Host to Device
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
	fprintf(stderr, "Kernel 1D Executed\n");
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
	DisplayMatrix1D("Convolution 1D Result", h_out1D, N);
	// Verification
	int *h_Verification1D = (int*)malloc(sizeArr);
	verify1Dresult(h_arr, h_conv, h_Verification1D);
	DisplayMatrix1D("Serial Code", h_Verification1D, N);
	for(int i = 0; i < N; i++) {
		if(h_out1D[i]!=h_Verification1D[i]) {
			fprintf(stderr, "Result Verification Failed\n");
			exit(EXIT_FAILURE);
		}
	}
	printf("Verification Completed \n\n");
	
	// Free Device Memory
	cudaFree(d_conv);
	cudaFree(d_arr);
	cudaFree(d_1DconvResult);
	// Free Host Memory
	free(h_arr);
	free(h_conv);
	free(h_out1D);
	free(h_Verification1D);

	/*
	 2D convolution 
	*/
	sizeArr = N*N*sizeof(float);
	float *h_2Darr = (float*)malloc(sizeArr);
	float *h_2Dmask = (float*)malloc(mask2D*sizeof(float));
	float *d_2Darr = NULL;
	err = cudaMalloc((void **)&d_2Darr, sizeArr);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to device array. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	float *d_2Dmask = NULL;
	err = cudaMalloc((void **)&d_2Darr, mask2D*sizeof(float));
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to 2Ddevice convolution array. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Filling up the values
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			h_2Darr[i*N + j] = (float)(rand()%10);
		}
	}
	// Filling up the 2D mask
	for(int i = 0; i < mask2D; i++) {
		h_2Dmask[i] = (i==mask2D/2)? 0.0 : 1.0/8;
	}
	//Display Elements
	DisplayMatrix2D("2D Matrix Initialized", h_2Darr, N);
	DisplayMatrix2D("2D mask Kernel", h_2Dmask, 3);
	// copying from Host to Device
	cudaMemcpy(d_2Darr, h_2Darr, sizeArr, cudaMemcpyHostToDevice);
	cudaMemcpy(d_2Dmask, h_2Dmask, mask2D*sizeof(float), cudaMemcpyHostToDevice);
	// Output Arrays
	float *h_2Dout = (float*)malloc(sizeArr);
	float *d_2dresult = NULL;
	err = cudaMalloc((void **)&d_2dresult, sizeArr);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to device array to store result computed. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Defining blocks
	gridSize = (int)ceil((float)N/(blockSize*blockSize));
	// Dimensions
	dim3 blocksPerGrid2D(gridSize);
	dim3 threadsPerBlock2D(1024);
	// Launching Kernel
	convolution2D<<<blocksPerGrid2D, threadsPerBlock2D>>>(d_2Darr, d_2Dmask, d_2dresult, N);
	fprintf(stderr, "Kernel 2D Executed\n");
	err = cudaGetLastError();
	if(err != cudaSuccess) {
		fprintf(stderr, "Failed to Launch 2D Convolution Kernel");
		exit(EXIT_FAILURE);
	}
	err = cudaMemcpy(h_2Dout, d_2dresult, sizeArr, cudaMemcpyDeviceToHost);
	if(err != cudaSuccess){
		fprintf(stderr, " Failed to copy from device memory to host memory for convolution 2D with error code %s\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("2D convolution Kernel Executed\n");
	DisplayMatrix2D("Convolution 2D Result", h_2Dout, N);
	// Reset the Device and Exit
	err = cudaDeviceReset();
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	printf("Done\n");
	return 0;
}

void DisplayMatrix1D(const char *inp, int *arr, int num) {
	printf("%s\n", inp);
	for(int i = 0; i < num; i++) {
		printf("%d \t", arr[i]);
	}
	printf("\n\n");
}

void verify1Dresult(int *arr, int *conv, int *h_Verification1D) {
	for(int i = 0; i < N; i++) {
		int k = M/2, convSum = 0, cnum = 0;
		for(int j = -k; j <= k; j++) {
			if(i+j >= 0 && i+j < N && cnum < M) {
				convSum += arr[i+j]*conv[cnum];
			}
			cnum++;
		}
		h_Verification1D[i] = convSum;
	}
}

void DisplayMatrix2D(const char *inp, float *arr, int num) {
	printf("%s\n", inp);
	for(int i = 0; i < num; i++) {
		for(int j = 0; j < num; j++) {
			printf("%.3f \t", arr[i*num + j]);
		}
		printf("\n");
	}
	printf("\n\n");
}
