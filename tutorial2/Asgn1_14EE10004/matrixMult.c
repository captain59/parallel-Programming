#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>

#define N 15

int main(int argc, char const *argv[])
{
	double arrayA[N][N], arrayB[N][N];
	FILE *fp;
	int i, j, k;
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			arrayA[i][j] = rand()%10;
			arrayB[i][j] = rand()%10;
		}
	}
	// Printing files
	fp = fopen("A.txt", "w");
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			fprintf(fp, "%lf  ", arrayA[i][j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	fp = fopen("B.txt", "w");
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			fprintf(fp, "%lf  ", arrayA[i][j]);
		}
		fprintf(fp, "\n");
	}
	fclose(fp);
	fp = fopen("A.txt", "r");
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			fscanf(fp, "%lf", &arrayA[i][j]);
		}
	}
	fclose(fp);
	fp = fopen("B.txt", "r");
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			fscanf(fp, "%lf", &arrayB[i][j]);
		}
	}
	fclose(fp);
	double out[N][N];
	fp = fopen("Output.txt", "w");
	for(i = 0; i < N; i++) for(j = 0; j < N; j++) out[i][j] = 0.0;
	double time = omp_get_wtime();
	#pragma omp parallel for schedule(static, 3) private(i, j, k) shared(arrayA, arrayB, out)
		for(i = 0; i < N; i++) {
			for(j = 0; j < N; j++) {
				for(k = 0; k < N; k++) {
					#pragma omp critical
					out[i][j] += arrayA[i][k]*arrayB[k][j];
				}
			}
		}
	fprintf(fp, "Critical Time Taken %lf\n", omp_get_wtime() - time);
	fprintf(fp, "Critical Output\n");	
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			fprintf(fp, "%lf   ", out[i][j]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n");
	// Without Critical
	for(i = 0; i < N; i++) for(j = 0; j < N; j++) out[i][j] = 0.0;
	time = omp_get_wtime();
	#pragma omp parallel for schedule(static, 3) private(i, j, k) shared(arrayA, arrayB, out)
		for(i = 0; i < N; i++) {
			for(j = 0; j < N; j++) {
				for(k = 0; k < N; k++) {
					out[i][j] += arrayA[i][k]*arrayB[k][j];
				}
			}
		}
	fprintf(fp, "Without Collapse Time Taken %lf\n", omp_get_wtime() - time);
	fprintf(fp, "Without Collapse Output\n");
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			fprintf(fp, "%lf   ", out[i][j]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n");
	// Using Collapse
	for(i = 0; i < N; i++) for(j = 0; j < N; j++) out[i][j] = 0.0;
	time = omp_get_wtime();
	#pragma omp parallel for schedule(static, 3) private(i, j, k) shared(arrayA, arrayB, out) collapse(2)
		for(i = 0; i < N; i++) {
			for(j = 0; j < N; j++) {
				for(k = 0; k < N; k++)
					out[i][j] += arrayA[i][k]*arrayB[k][j];
			}
		}
	fprintf(fp, "Collapse Time Taken %lf\n", omp_get_wtime() - time);
	fprintf(fp, "Collapse Output\n");
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			fprintf(fp, "%lf   ", out[i][j]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n");
	//Using Reduction
	for(i = 0; i < N; i++) for(j = 0; j < N; j++) out[i][j] = 0.0;
	time = omp_get_wtime();
	double matSum = 0;
	#pragma omp parallel for schedule(static, 3) private(i, j, k) shared(arrayA, arrayB, out) reduction(+ : matSum)
		for(i = 0; i < N; i++) {
			for(j = 0; j < N; j ++) {
				matSum = 0;
				for(k = 0; k < N; k++)
					matSum += arrayA[i][k]*arrayB[k][j];
				out[i][j] = matSum;
			}
		}
	fprintf(fp, "Reduction Time Taken %lf\n", omp_get_wtime() - time);
	fprintf(fp, "Reduction Output\n");
	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			fprintf(fp, "%lf   ", out[i][j]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n\n");
	fclose(fp);
	return 0;
}
