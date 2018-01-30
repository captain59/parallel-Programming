#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void arradd(const int *md, const int *nd, int *pd, int size){
	int myid = blockDim.x*blockIdx.x + threadIdx.x;
	if(myid < size)
		pd[myid] = md[myid] + nd[myid];
}
int main(){
	int num = 200;
	size_t size = 200*sizeof(int);
	int i = 0;
	int *m, *n, *p;
	//Allocating memory on CPU
	m = (int*)malloc(size);
	n = (int*)malloc(size);
	p = (int*)malloc(size);
	for(i=0; i< num; i++)
		m[i]=i, n[i]=i/2, p[i]=0;

	// Allocating memery on the Gpu
	int *md, *nd, *pd;
	cudaMalloc((void **)&md, size);
	// (destination, sources, n.o of bytes, direction)
	cudaMemcpy(md, m, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **)&nd, size);
	cudaMemcpy(nd, n, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **)&pd, size);
	// no need to allocate size as addition will generate which transfer from gpu to cpu
	int blocksize = 1024;
	int gridsize = (int)ceil((float)num/blocksize);
	arradd<<<gridsize, blocksize>>>(md, nd, pd, num);

	cudaMemcpy(p, pd, size, cudaMemcpyDeviceToHost);
	cudaFree(md);
	cudaFree(nd);
	cudaFree(pd);
	for(i=0; i < num; i++)
		printf("%d + %d = %d\n", m[i], n[i], p[i]);
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
