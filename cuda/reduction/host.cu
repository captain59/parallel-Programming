#include "headers.h"

#define N 4
#define blockSize 1024

void DisplayMatrixRowMajor(const char *inp, int *arr, int noElements) {
	printf("%s\n", inp);
	for(int i = 0; i < noElements; i+=4) {
		printf("%d \t %d \t %d \t %d", arr[i], arr[i+1], arr[i+2], arr[i+3]);
		printf("\n\n");
	}
}

void DisplayOutputMatrixRowMajor(const char *inp, int *arr) {
	printf("%s\n", inp);
	int i = 0;
	printf("%d \t %d \t %d \t %d", arr[i], arr[i+1], arr[i+2], arr[i+3]);
	printf("\n\n");
}

int main(int argc, char const *argv[])
{
	// Error Code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;	
	int noElements = 4*N;
	// taking ceil
	int gridSize = (int)ceil((float)noElements/(blockSize));
	size_t size = noElements*sizeof(int);
	printf("[Operations on Kernel on %d elements] \n", noElements);
	// Host Matrix
	int *h_arr = (int*)malloc(size);
	// Creating Random Matrix
	for(int i = 0; i < noElements; i++) {
		h_arr[i] = rand()%10;
	}
	// Display initial Matrix
	DisplayMatrixRowMajor("Initial Matrix", h_arr, noElements);
	// Device Matric
	int *d_arr = NULL;
	err = cudaMalloc((void **)&d_arr, size);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to device array. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
	// Output Matrix
	int *d_out = NULL;

	err = cudaMalloc((void **)&d_out, size);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to device Output array. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	int *d_out2 = NULL;

	err = cudaMalloc((void **)&d_out2, size);
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to allocates size to device Output array. Error code %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	// Dimensions
	dim3 blocksPerGrid(gridSize);
	dim3 threadsPerBlock(blockSize);

	size_t shmsz = blockSize*sizeof(int);
	ReduceRowMajor<<< blocksPerGrid, threadsPerBlock , shmsz >>>(d_arr, d_out, noElements);

	int *h_out = (int*)malloc(shmz);
	cudaMemcpy(h_out, d_out2, shmz, cudaMemcpyDeviceToHost);

	// Display The Matrix
	DisplayOutputMatrixRowMajor("Row Major Sum Matrix", h_out);
	free(h_arr);
	free(h_out);
	cudaFree(d_arr);
	cudaFree(d_out);
	cudaFree(d_out2);
	// Reset the Device and Exit
	err = cudaDeviceReset();
	if(err != cudaSuccess){
		fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
	}
	printf("Done\n");
	return 0;
}
