#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include "ptools1.h"

#define TAG_ONE 1
#define TAG_TWO 2
#define TAG_THREE 3

int main(int argc, char *argv[]){
    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int D = MATRIX_DIMENSION;
    int K = TOP_K;

    double t_start;
    double t_end;
    double t_total;
    float diff;

    int rank;
    int size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int num_rows = N / size;
    int record_rows = M / size;
    float *local_record = (float*)malloc(num_rows*M*sizeof(float));
    float *row = (float*)malloc(N*sizeof(float));
    int *topK;
    int offSet;

    float *A, *B;
    float *buffer = (float*)malloc(num_rows*D*sizeof(float));
    float *A_copy = (float*)malloc(M*D*sizeof(float));
    float *A_copy2 = (float*)malloc(M*D*sizeof(float));
    float *record = (float*)malloc(M*N*sizeof(float));
    MPI_Status status;

    if(rank == 0){
        A = generateMatrix(M, D);
        B = generateMatrix(N, D);
        record = (float*)malloc(M*N*sizeof(float));
        t_start = MPI_Wtime();
        for (int i=1;i<size;i++) {
            MPI_Send(A,M*D,MPI_FLOAT,i,TAG_ONE,MPI_COMM_WORLD);
        }
        for (int i=1; i<size; i++) {
            MPI_Send(B+(i-1)*num_rows*D,num_rows*D,MPI_FLOAT,i,TAG_TWO,MPI_COMM_WORLD);
        }

        for (int i=0; i<M; i++) {
            for (int j=(size-1)*num_rows; j<N; j++) {
                //record[i*N+j] = getEuclideanDistance(A, i, B, j, D);
                record[i*N+j] = fvec_L2sqr_sse(A+i*D, B+j*D, D-1);
            }
        }

        for (int i=1; i<size;i++) {
            MPI_Recv(local_record,M*num_rows,MPI_FLOAT,i,TAG_THREE,MPI_COMM_WORLD,&status);
            for (int j=0; j<M; j++) {
                offSet = j*N + (i-1)* num_rows;
                for (int k=0;k<num_rows;k++) {
                    record[offSet+k] = local_record[j*num_rows+k];
                }
            }
        }

        labelMatrices(A, record, B, M, N, D, K);

    } else {
        MPI_Recv(A_copy,M*D,MPI_FLOAT,0,TAG_ONE,MPI_COMM_WORLD, &status);
        MPI_Recv(buffer,num_rows*D,MPI_FLOAT,0,TAG_TWO,MPI_COMM_WORLD, &status);
        for(int i=0; i<M; i++) {
            for (int j = 0; j < num_rows; j++) {
                //local_record[i*num_rows+j] = getEuclideanDistance(A_copy, i, buffer, j, D);
                local_record[i*num_rows+j] = fvec_L2sqr_sse(A_copy+i*D, buffer+j*D, D-1);
            }
        }
        MPI_Send(local_record,M*num_rows,MPI_FLOAT,0,TAG_THREE,MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0){
        t_end = MPI_Wtime();
        t_total = t_end - t_start;
        printf("KNN MPI total seconds: %lf\n", t_total);
    }

    MPI_Finalize();

    if(rank == 0){
        memcpy(A_copy2, A, M*D*sizeof(float));
        serialKNN1(A_copy2, B, M, N, D, K);
        validateLables(A, A_copy2, M, D);
        free(B);
        free(A);
        free(A_copy2);
    }

    free(buffer);
    free(record);

    return 0;
}

