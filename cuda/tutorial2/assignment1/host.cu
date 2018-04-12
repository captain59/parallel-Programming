#include "headers.h"

#define N 10
#define blockSize 32

void  DisplayMatrix(const char *inp, int *arr) {
	printf("%s\n", inp);
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			printf("%d \t", arr[i*N + j]);
		}
		printf("\n");
	}
	printf("\n\n");
}

int main(int argc, char const *argv[])
{
	// Error Code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;	
	// taking ceil
	int gridSize = (int)ceil((float)N/(blockSize*blockSize));
	size_t size = N*N*sizeof(int);
	printf("[Operations on Kernel on %d elements] \n", N);
	// Host Matrix
	int *h_arr = (int*)malloc(size);
	// Creating Random Matrix
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			h_arr[i*N + j] = rand()%100;
		}
	}
	// Display initial Matrix
	DisplayMatrix("Initial Matrix", h_arr);
	// Device Matric
	int *d_arr = NULL;
	err = cudaMalloc((void **)&d_arr, size*size);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to device array. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
	// Dimensions
	dim3 blocksPerGrid(gridSize);
	dim3 threadsPerBlock(32, 32);
	MatrixOp<<< blocksPerGrid, threadsPerBlock >>>(d_arr, N);
	
	int *h_out = (int*)malloc(size);
	cudaMemcpy(h_out, d_arr, size, cudaMemcpyDeviceToHost);
	// Display The Matrix
	DisplayMatrix("Transformed Matrix", h_out);
	free(h_arr);
	free(h_out);
	cudaFree(d_arr);
	// Reset the Device and Exit
	err = cudaDeviceReset();
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	printf("Done\n");
	return 0;
}
