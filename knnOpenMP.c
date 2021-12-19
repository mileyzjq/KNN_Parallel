#include "ptools1.h"
#include <string.h>
#include "omp.h"

int N, M, D, K, NUM_THREADS;
float *A, *B;


void knnOpenMP(float *A, float *B, float *record, int M, int N, int D) {
    int i, j;
#pragma omp parallel shared(A, B, record) private(i, j)
    {
#pragma omp for
        for (j = 0; j < N; j++) {
            for (i = 0; i < M; i++) {
                record[i * N + j] = fvec_L2sqr_sse(A+i*D, B+j*D, D-1);
                //record[i * N + j] = getEuclideanDistance(A, i, B, j, D);
            }
        }

    }
}

void labelOpenMP(float *A, float *record, float *B, int M, int N, int D, int K) {
    float *row = (float*)malloc(N*sizeof(float));
    int* topK;
    int i;
#pragma omp for
    for(i=0; i<M; i++) {
        memcpy(row, &record[i*N], N * sizeof(float));
        topK = findTopK(row, N, K);
        A[i*D+D-1] = labelVectorArray(topK, K, B, N, D);
    }
}

int main(int argc, char *argv[]) {
    double duration;
    N = atoi(argv[1]);
    M = atoi(argv[2]);
    D = MATRIX_DIMENSION;
    K = TOP_K;
    NUM_THREADS = atoi(argv[3]);
    omp_set_num_threads(NUM_THREADS);
    struct timeval start1, stop1;
    double start, stop, stop2;
    A = generateMatrix(M, D);
    B = generateMatrix(N, D);
    float *A_copy = (float*)malloc(M*D*sizeof(float));
    memcpy(A_copy, A, M*D*sizeof(float));
    float diff;
    int* topK;
    int i, j, tid, Nthrds, k;
    float* record = (float*)malloc(M*N*sizeof(float));

    start = omp_get_wtime();
    knnOpenMP(A, B, record, M, N, D);
    stop2 = omp_get_wtime();
    //printf("KNN OpenMP takes %lf seconds\n", stop2-start);
    //labelMatrices(A, record, B, M, N, D, K);
    labelOpenMP(A, record, B, M, N, D, K);

    stop = omp_get_wtime();
    printf("KNN OpenMP takes %lf seconds\n", stop-start);  

    gettimeofday(&start1, 0);
    serialKNN1(A_copy, B, M, N, D, K);
    gettimeofday(&stop1, 0);
    duration = (stop1.tv_sec - start1.tv_sec) + (stop1.tv_usec - start1.tv_usec) * 1e-6;
    printf("Serial KNN takes %lf seconds \n", duration);

    validateLables(A, A_copy, M, D);

    free(A);
    free(B);
    return 0;
}

