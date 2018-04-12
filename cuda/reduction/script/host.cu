#include "headers.h"

#define N 2048
#define blockSize 512

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

void serialCode(int *arr, int noElements) {
	int i = 0;
	int *sum = (int*)malloc(5*sizeof(int));
	for(i = 0; i < 5; i++)
		sum[i] = 0;
	for(i = 0; i < noElements; i+=4) {
		sum[0] += arr[i];
		sum[1] += arr[i+1];
		sum[2] += arr[i+2];
		sum[3] += arr[i+3];
	}
	i = 0;
	printf("Serial Code Output \n");
	printf("%d \t %d \t %d \t %d", arr[i], arr[i+1], arr[i+2], arr[i+3]);
	printf("\n\n");
}

int main(int argc, char const *argv[])
{
	// Declaring Time Variables
	clock_t start, end;
	double gpu_time;
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
	//DisplayMatrixRowMajor("Initial Matrix", h_arr, noElements);
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
	size_t shmz = gridSize*4*sizeof(int);

	// start Time
	start = clock();
	ReduceRowMajor5<<< blocksPerGrid, threadsPerBlock , shmsz >>>(d_arr, d_out, noElements);

	
	ReduceRowMajor5<<< 1, blocksPerGrid , shmz >>>(d_out, d_out2, gridSize*4);
	// end Time
	end = clock();
	int *h_out = (int*)malloc(shmz);
	cudaMemcpy(h_out, d_out2, shmz, cudaMemcpyDeviceToHost);
	// serial code

	serialCode(h_out, noElements);
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
	gpu_time = ((double)(end - start))/CLOCKS_PER_SEC;
	printf("Time Taken %lf \n", gpu_time);
	printf("Done\n");
	return 0;
}



