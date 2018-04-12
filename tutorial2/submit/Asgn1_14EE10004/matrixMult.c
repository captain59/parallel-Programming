#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>

#define N 15

void saveMatrix(const char *fileName, double *out[]) {
	FILE *fp = fopen(fileName, "w");
	fprintf(fp, "The Matrix: %s \n", fileName);
	int i, j;
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			fprintf(fp, "%lf \t", out[i][j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
}


void **obtainRandomMatrix(const char *fileName) {
	double **retMat;
	int i, j;
	retMat = (double **)malloc(sizeof(double *)*N);
	if(retMat == NULL) {
		perror("Memory Allocation Failed for Ns.");
		exit(EXIT_FAILURE);
	}
	for(i = 0; i < N; i++) {
		retMat[i] = (double *)malloc(sizeof(double)*N);
		if(retMat[i] == NULL) {
			perror("Memory Allocation Failed for Numns");
			exit(EXIT_FAILURE);
		}
	}
	for(i = 0; i < N; i++) {
		for(j = 0; j< N; j++) {
			retMat[i][j] = rand()%100;
		}
	}
	saveMatrix(fileName, retMat);
}

double **initToZero() {
	double **retMat;
	int i, j;
	retMat = (double **)malloc(sizeof(double *)*N);
	if(retMat == NULL) {
		perror("Memory Allocation Failed for Ns.");
		exit(EXIT_FAILURE);
	}
	for(i = 0; i < N; i++) {
		retMat[i] = (double *)malloc(sizeof(double)*N);
		if(retMat[i] == NULL) {
			perror("Memory Allocation Failed for Numns");
			exit(EXIT_FAILURE);
		}
	}
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++)
			retMat[i][j] = 0;
	}
	return retMat;
}

// read matrix
double **readMatrix(const char *fileName) {
	double **mat = initToZero();
	FILE *fp = fopen(fileName, "r");
	int i, j;
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			fscanf(fp, "%lf", &mat[i][j]);
		}
	}
	return mat;
}

int main(int argc, char const *argv[])
{
	obtainRandomMatrix("A.txt");
	obtainRandomMatrix("B.txt");
	// readMatrix
	double **arrayA = readMatrix("A.txt");
	double **arrayB = readMatrix("B.txt");
	int chunkSize = 15;
	// critical
	double **output_critical = initToZero();
	int i, j, k;
	double waittime = omp_get_wtime();
	#pragma omp parallel for schedule(dynamic, chunkSize)
	for(i = 0; i < N; i++) {
		#pragma omp parallel for schedule(dynamic, chunkSize)
		for(j = 0; j < N; j++) {
			for(k = 0; k < N; k++) {
				#pragma omp critical 
				{
					output_critical[i][j] += arrayA[i][k]*arrayB[k][j];
				}	
			}
		}	
	}
	double criticalTime = omp_get_wtime() - waittime;
	// collapse method
	double **output_collapse = initToZero();
	waittime = omp_get_wtime();

	#pragma omp parallel shared(arrayA, arrayB, output_critical)
		#pragma omp for collapse(2)
			for(i = 0; i < N; i++) for(j = 0; j < N; j++) for(k = 0; k < N; k++) output_collapse[i][j] += arrayA[i][k]*arrayB[j][k];
	double collapseTime = omp_get_wtime() - waittime;
	// reduction method
	waittime = omp_get_wtime();
	double **output_reduction = initToZero();
	#pragma omp parallel for schedule(dynamic, chunkSize)
	for(i = 0; i < N; i++) {
		#pragma omp parallel for schedule(dynamic, chunkSize)
		for(j = 0; j < N; j++) {
			double multSum = 0;
			#pragma omp parallel for reduction(+ : multSum), schedule(dynamic, chunkSize)
			for(k = 0; k < N; k++)
				multSum += arrayA[i][k]*arrayB[k][j];
			output_reduction[i][j] = multSum;
		}
	}
	double reductionTime = omp_get_wtime() - waittime;
	saveMatrix("Critical_Result.txt", output_critical);
	saveMatrix("collapse_Result.txt", output_collapse);
	saveMatrix("Reduction_Result.txt", output_reduction);
	// Saving Time Taken
	FILE *fp = fopen("Time.txt", "w");
	fprintf(fp, "Critical Time: %lf\n", criticalTime);
	fprintf(fp, "Collapse Time: %lf\n", collapseTime);
	fprintf(fp, "Reduction Time: %lf\n", reductionTime);
	fclose(fp);
	return 0;
}