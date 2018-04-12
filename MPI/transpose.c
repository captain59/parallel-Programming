#include <stdio.h>
#include <mpi.h>

#define NPROCESSORS 16

#define N 10000

void Transpose(float *a) {
	int i, j, k = 0;
	for(i = 0; i < N; i++) {
		k = i + 1 + k;
		for(j = i+1; j < N; j++) {
			float t = a[k];
			a[k++] = a[(N+1)*(i+1) - 1 + N*(j - i + 1)];
			a[(N+1)*(i+1) - 1 + N*(j-i-1)] = t;
		}
	}
}

int main(int argc, char *argv[])
{
	float A[N][N];
	float B[N][N];
	int i, j, nprocs, rank;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if(rank==0) {
		printf("Matrix Transpose with %d processor\n", NPROCESSORS);
	}
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++)
			A[i][j] = (i*j)/10000;
	}
	MPI_Alltoall(&A[0][0], N*N, MPI_FLOAT, &B[0][0], N*N, MPI_FLOAT, MPI_COMM_WORLD);
	for(i = 0; i< NPROCESSORS; i++)
		Transpose(&B[i*N][0]);
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			if(B[i][j] != i*(j + rank*N)/10000) {
				printf("Error\n");
				MPI_Abort(MPI_COMM_WORLD, 1);
				return 1;
			}
		}
	}
	if(rank == 0)
		printf("Correct\n");
	MPI_Finalize();
	return 0;
}
