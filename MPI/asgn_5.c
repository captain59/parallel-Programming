
#include <stdio.h>
#include <mpi.h>


#define HSIZE 10000
#define VSIZE 625
#define HIST_SIZE 250
#define NPROC 16

int main (int argc, char *argv[])
{
  int a[HIST_SIZE];
  int b[HIST_SIZE];
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

  for (i = 0; i < HIST_SIZE; i++)
    {
      a[i]=0;
      b[i]=0;
    }
  for (i = 0; i < HSIZE; i++)
    for (j = 0; j < VSIZE; j++)
      a[(int)(i+(j+(rank*VSIZE)))/80]++;

  MPI_Reduce(a, b ,HIST_SIZE, MPI_INT, MPI_SUM, 0,MPI_COMM_WORLD);
  if(rank==0){
    
    for (i = 0; i < HIST_SIZE; i++)
      {
        a[i]=0;
      }
    for (i = 0; i < HSIZE; i++)
      for (j = 0; j < HSIZE; j++)
        a[(int)(i+j)/80]++;
    for (i = 0; i < HIST_SIZE; i++)
      {
        if (b[i] != a[i] )
          {
          printf ("Error at place %d , calculated = %d , correct = %d ",i,b[i],a[i]);
          MPI_Abort (MPI_COMM_WORLD,1);
          return 1;
          }
      }
  }
  if (rank == 0)
    printf ("\nHistogram seems ok\n");

  MPI_Finalize ();
  return 0;
}
