#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>

#define LO(num) ((num) & 0x000000FF) 

typedef struct _PGMData {
	int row, col, maxValue;
	int **mat;
} pgmdata;

int **allocate_size(int row, int col) {
	int **retMat;
	int i;
	retMat = (int **)malloc(sizeof(int *)*row);
	if(retMat == NULL) {
		perror("Memory Allocation Failed for rows.");
		exit(EXIT_FAILURE);
	}
	for(i = 0; i < row; i++) {
		retMat[i] = (int *)malloc(sizeof(int)*col);
		if(retMat[i] == NULL) {
			perror("Memory Allocation Failed for columns");
			exit(EXIT_FAILURE);
		}
	}
	return retMat;
}

void skipComments(FILE *fp) {
	int ch;
	char line[2048];
	while((ch == fgetc(fp)) != EOF && isspace(ch)) { ; }
	if(ch == '#') {
		fgets(line, sizeof(line), fp);
		skipComments(fp);
	}
	else {
		fseek(fp, -1, SEEK_CUR);
	}
}

/*
Reading pgmdata file
*/
pgmdata* readPGMData(const char *fileName) {
	pgmdata *data = (pgmdata*)malloc(sizeof(pgmdata));
	FILE *pgmFile;
	char version[3];
	int i, j, lo, hi;
	// open the file
	pgmFile = fopen(fileName, "rb");
	if(pgmFile == NULL) {
		perror("Unable to open file to read");
		exit(EXIT_FAILURE);
	}
	fgets(version, sizeof(version), pgmFile);
	if(strcmp(version, "P5")) {
		fprintf(stderr, "Wrong File Type\n");
		exit(EXIT_FAILURE);
	}
	skipComments(pgmFile);
	fscanf(pgmFile, "%d", &data->col);
	skipComments(pgmFile);
	fscanf(pgmFile, "%d", &data->row);
	skipComments(pgmFile);
	fscanf(pgmFile, "%d", &data->maxValue);
	fgetc(pgmFile);
	data->mat = allocate_size(data->row, data->col);
	for(i = 0; i < data->row; i++) {
		for(j = 0; j < data->col; j++) {
			lo = fgetc(pgmFile);
			data->mat[i][j] = lo;
		}
	}
	fclose(pgmFile);
	return data;
}
// Writing pgmdata file
void writePGM(const char *fileName, const pgmdata *data, int *output[]) {
	FILE *pgmFile;
	int i, j, hi, lo;
	pgmFile = fopen(fileName, "wb");
	if(pgmFile == NULL) {
		perror("Unable to open file to write");
		exit(EXIT_FAILURE);
	}
	fprintf(pgmFile, "P5 ");
    fprintf(pgmFile, "%d %d ", data->col, data->row);
    fprintf(pgmFile, "%d ", data->maxValue);
    for (i = 0; i < data->row; ++i) {
        for (j = 0; j < data->col; ++j) {
            lo = LO(output[i][j]);
            fputc(lo, pgmFile);
        }
    }
    fclose(pgmFile);
}

void writeHistogram(const char *fileName, const int *histogram, const int maxValue) {
	FILE *fp = fopen(fileName, "w");
	int i, j;
	for(i = 0; i < maxValue; i++) {
		fprintf(fp, "%d :==>> %d \n", i, histogram[i]);
	}
	fclose(fp);
}
int main(int argc, char const *argv[]) {
	int i, j;
	pgmdata *data = readPGMData("Julia_IIM_6_circle.pgm");
	int *histogram = (int*)calloc(data->maxValue + 1, sizeof(int));
	int **output = (int **)malloc(data->row*sizeof(int*));
	for(i = 0; i < data->row; i++)
		output[i] = (int*)malloc(data->col*sizeof(int));
	int temp = 2;
	#pragma omp parallel for schedule(dynamic, 5) private(i, j)
	for(i = 0; i < data->row; i++) {
		for(j = 0; j < data->col; j++) {
			output[i][j] = data->mat[i][j];
			#pragma omp atomic
			histogram[data->mat[i][j]]++;
		}
	}
	// Writing Histogram Complete
	writeHistogram("histogram.txt", histogram, data->maxValue + 1);
	// Averaging
	#pragma omp parallel for schedule(dynamic, 5)
	for(i = 0; i < data->row-9; i++) {
		for(j = 0; j < data->col-9; j++) {
			int kernelSum = 0, k, l;
			#pragma omp parallel for reduction(+: kernelSum)
				for(k = i; k < i+10; k++) {
					for(l = j; l < j+10; l++) {
						kernelSum += data->mat[k][l];
					}
				}
			if(kernelSum < 2000) {
					for(k = i; k < i+10; k++) {
						for(l = j; l < j+10; l++) {
							output[k][l] = 0;
						}
					}
			}
		}
	}
	writePGM("outputImage.pgm", data, output);
	return 0;
}
