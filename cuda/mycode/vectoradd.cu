#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void arradd(const int *md, const int *nd, const int *pd, int size){
	int myid = blockDim.x * blockIdx.x + threadIdx.x;
	p[myid] = md[myid] + nd[myid];
}
int main(){
	int size = 200*sizeof(int);
	int i = 0;
	int *m, *n, *p;
	//Allocating memory on CPU
	m = (int*)malloc(size);
	n = (int*)malloc(size);
	p = (int*)malloc(size);
	for(i=0; i<200; i++)
		m[i]=i, n[i]=i, p[i]=0;

	// Allocating memery on the Gpu
	int *md, *nd, *pd;
	cudaMalloc(&md, size);
	// (destination, sources, n.o of bytes, direction)
	cudaMemcpy(md, m, size, cudaMemcpyHostToDevice);

	cudaMalloc(&nd, size);
	cudaMemcpy(nd, n, size, cudaMemcpyHostToDevice);

	cudaMalloc(&pd, size);
	// no need to allocate size as addition will generate which transfer from gpu to cpu
	cudaMemcpy(pd, size);

	dim3 DimGrid(1, 1);
	dim3 DimBlock(200, 1);

	arradd<<<DimGrid, DimBlock>>(md, nd, pd, size);

	cudaMemcpy(p, pd, size, cudaMemcpyDeviceToHost);
	for(int i=0;i < 200; i++)
		printf("%d\n", p[i]);
	cudaFree(md);
	cudaFree(nd);
	cudaFree(pd);
	free(m);
	free(n);
	free(p);
	// Reser the Device and exit
	cudaError_t err = cudaDeviceReset();
	if( err != cudaSuccess){
		printf("Failed to deinitialize the device error %s \n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	printf("Done\n");
	return 0;
}