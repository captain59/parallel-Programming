
#include <stdio.h>
#include <mpi.h>


#define HSIZE 10000
#define VSIZE 625
#define NPROC 16

void T (float *a){
  int i, j,k=0;
  for (i = 0; i < VSIZE; i++)
    {
      k = i + 1 + k;
      for (j = i+1; j < VSIZE; j++)
      {
        float t = a[k];
        a[k++] = a[(VSIZE+1)*(i+1) - 1 + (VSIZE)*(j-i-1)];
        a[(VSIZE+1)*(i+1) - 1 + (VSIZE)*(j-i-1)] = t;
      }
    }
    
}

int main (int argc, char *argv[])
{
  float a[HSIZE][VSIZE];
  float b[HSIZE][VSIZE];
  int i, j, nprocs, rank;

  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  if(rank == 0)
  {
    printf("Transposing a %dX%d matrix, divided among %d processors\n",HSIZE,HSIZE,NPROC);
  }
  if (nprocs != NPROC)
    {
      if (rank == 0)
	      printf ("Error, number of processes must be %d\n",NPROC);
      MPI_Finalize ();
      return 1;
    }

  for (i = 0; i < HSIZE; i++)
    for (j = 0; j < VSIZE; j++)
      a[i][j] = i*(j+(rank*VSIZE))/10000;

  MPI_Alltoall (&a[0][0],VSIZE * VSIZE,MPI_FLOAT,&b[0][0],VSIZE * VSIZE,MPI_FLOAT,MPI_COMM_WORLD);


  for (i = 0; i < NPROC; i++)T(&b[i * VSIZE][0]);
  for (i = 0; i < HSIZE; i++)
    for (j = 0; j < VSIZE; j++)
      {
		if (b[i][j] != i*(j+(rank*VSIZE))/10000 )
		  {
			printf ("process %d found b[%d][%d] = %f, but %f was expected\n",rank, i, j, b[i][j], (float) i*(j+(rank*VSIZE))/10000);
			MPI_Abort (MPI_COMM_WORLD,1);
			return 1;
		  }
      }
  if (rank == 0)
    printf ("Transpose seems ok\n");

  MPI_Finalize ();
  return 0;
}
