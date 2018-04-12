#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>


#define LENGTH 100
#define WIDTH 200
#define DEPTH 400
#define G 6.67e-11
#define NUM 1000
#define HOUR 3600
#define DELTA 0.005

typedef struct vector {
	double x;
	double y;
	double z;
} vec;

double positionInRange(float X) {
	return (X*rand())/RAND_MAX;
}
void initPosition(vec *st) {
	st->x = positionInRange(LENGTH);
	st->y = positionInRange(WIDTH);
	st->z = positionInRange(DEPTH);
}
void initZero(vec *st) {
	st->x = 0;
	st->y = 0;
	st->z = 0;
}
vec doSubtract(const vec a, const vec b) {
	vec ret;
	ret.x = a.x - b.x;
	ret.y = a.y - b.y;
	ret.z = a.z - b.z;
	return ret;
}
vec doMultiply(const double mul, const vec a) {
	vec ret;
	ret.x = mul*a.x;
	ret.y = mul*a.y;
	ret.z = mul*a.z;
	return ret;
}
vec doAdd(const vec a, const vec b) {
	vec ret;
	ret.x = a.x + b.x;
	ret.y = a.y + b.y;
	ret.z = a.z + b.z;
	return ret;
}
double getDistance(const vec a, const vec b) {
	vec diff = doSubtract(a, b);
	return sqrt(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
}
int main(int argc, char const *argv[])
{
	FILE *log, *trajectory;
	log = fopen("log.txt", "w");
	trajectory = fopen("trajectory.txt", "w");
	fprintf(trajectory, "NUMBER OF Objects = %d HOUR = %d DELTA = %lf\n", NUM, HOUR, DELTA);
	fprintf(trajectory, "X co-ordinate \t Y co-ordinate \t Z co-ordinate\n");
	fprintf(log, "NUM = %d HOUR = %D DELTA = %lf\n", NUM, HOUR, DELTA);
	fprintf(log, "Number of Objects = %d\n", NUM);
	fprintf(log, "TIME = %d\n", HOUR);
	fprintf(log, "DELTA INTERVAL = %lf\n", DELTA);
	fprintf(log, "Number of Threads = %d\n", omp_get_max_threads());
	double *radius = (double*)malloc((NUM+1)*sizeof(double));
	double *mass = (double*)malloc((NUM+1)*sizeof(double));
	vec *position = (vec*)malloc((NUM+1)*sizeof(vec));
	vec *velocity = (vec*)malloc((NUM+1)*sizeof(vec));
	vec *currentacceleration = (vec*)malloc((NUM+1)*sizeof(vec));
	vec *previousacceleration = (vec*)malloc((NUM+1)*sizeof(vec));
	// FIlling with values
	for(int i = 1; i <= NUM; i++) {
		radius[i] = 0.5;
		mass[i] = 1;
		initPosition(&position[i]);
		initZero(&velocity[i]);
		initZero(&currentacceleration[i]);
		initZero(&previousacceleration[i]);
	}
	int itterations = (int)ceil(HOUR/DELTA);
	fprintf(log, "Total Itterations: %d\n", itterations);
	double totalTime = omp_get_wtime();

	for(int t = 1; t <= itterations; t++) {
		double itterationTime = omp_get_wtime();
		int collisions = 0;
		#pragma omp parallel for 
		for(int i = 1; i<= NUM; i++) {
			initZero(&currentacceleration[i]);
			for(int j = 1; j <= NUM; j++) {
				if(i==j)
					continue;
				double distance = getDistance(position[j], position[i]);
				if( distance < radius[i] + radius[j]) {
					collisions++;
				}
				vec collisionacceleration = doMultiply(G*mass[j]/(distance*distance*distance), doSubtract(position[j], position[i]));
				currentacceleration[i] = doAdd(currentacceleration[i], collisionacceleration);
				// criterion for collision			
			}
		}
		//updating the position
		#pragma omp parallel for
		for(int i = 1; i <= NUM; i++) {
			vec firstTerm = doMultiply(DELTA, velocity[i]);
			vec secondTerm = doMultiply(0.5*DELTA*DELTA, currentacceleration[i]);
			position[i] = doAdd(position[i], doAdd(firstTerm, secondTerm));
		}
		// updating the velocity
		#pragma omp parallel for 
		for(int i = 1; i <= NUM; i++) {
			vec term = doMultiply(0.5*DELTA, doAdd(currentacceleration[i], previousacceleration[i]));
			velocity[i] = doAdd(velocity[i], term);
			// checking for collision against wall
			if((position[i].x + radius[i]) > LENGTH || (position[i].x < radius[i]))
				velocity[i].x = -1*velocity[i].x;
			if((position[i].y + radius[i]) > WIDTH || (position[i].y < radius[i]))
				velocity[i].y = -1*velocity[i].y;
			if((position[i].z + radius[i]) > DEPTH || (position[i].z < radius[i]))
				velocity[i].z = -1*velocity[i].z;
		}
		// updating previous acceleration
		#pragma omp parallel for
		for(int i=1; i <= NUM; i++) {
			previousacceleration[i] = currentacceleration[i];
		}

		//printing results
		if(t%1000 == 0) {
			printf("Printing File \n");
			fprintf(log, "Itteration: %d\n", t);
			fprintf(log, "Time Required for each step: %lf\n", omp_get_wtime() - itterationTime);
			//fprintf(log, "Collisions occured %d\n", collisions/2);
			fprintf(trajectory, "Itteration: %d\n", t);
			for(int i = 1; i <= NUM; i++) {
				fprintf(trajectory, "%lf \t %lf \t %lf\n", position[i].x, position[i].y, position[i].z);
			}
			fprintf(trajectory, "TERMINATE\n");
		}		
	}
	fprintf(log, "Time taken for %d itterations is %lf\n", itterations, omp_get_wtime() - totalTime);
	fclose(log);
	fclose(trajectory);
	free(radius);
	free(mass);
	free(position);
	free(velocity);
	free(currentacceleration);
	free(previousacceleration);
	return 0;
}
