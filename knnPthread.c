#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include "barrier.h"
#include <time.h>
#include <sys/time.h>
#include "ptools1.h"
#include "barrier.h"

void setThreads(float* A, float* B);
void *knnPthreads(void *args);

struct Matrix_t {
    int threadId;
    pthread_barrier_t *barrier;
    float *record;
};

int N, M, D, K;
int NUM_THREADS;
pthread_mutex_t lock;
float *A, *B;
float *record_total;

int main(int argc, char *argv[]){
    double duration;
    N = atoi(argv[1]);
    M = atoi(argv[2]);
    D = MATRIX_DIMENSION;
    K = TOP_K;
    NUM_THREADS = atoi(argv[3]);
    struct timeval start, stop;
    A = generateMatrix(M, D);
    B = generateMatrix(N, D);
    float *A_copy = (float*)malloc(M*D*sizeof(float));
    memcpy(A_copy, A, M*D*sizeof(float));
    record_total = (float*)malloc(M*N*sizeof(float));

    gettimeofday(&start, 0);
    setThreads(A, B);
    gettimeofday(&stop, 0);
    duration = (stop.tv_sec - start.tv_sec) + (stop.tv_usec - start.tv_usec) * 1e-6;
    printf("Pthread KNN takes %lf seconds \n", duration);

    serialKNN1(A_copy, B, M, N, D, K);
    validateLables(A, A_copy, M, D);

    free(A);
    free(B);
    return 0;
}

void setThreads(float* A, float* B) {
    pthread_t threads[NUM_THREADS];
    pthread_barrier_t barrier;
    struct Matrix_t matrixThreads[NUM_THREADS];
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);

    for(int i=0; i<NUM_THREADS; i++) {
        matrixThreads[i].threadId = i;
        matrixThreads[i].barrier = &barrier;
        matrixThreads[i].record = (float*)malloc(N*sizeof(float));
        if(pthread_create(&threads[i], NULL, knnPthreads, (void*)&matrixThreads[i]) != 0) {
            perror("ERROR: fail to create thread...\n");
        }
    }
    for(int i=0; i<NUM_THREADS; i++) {
        if(pthread_join( threads[i], NULL) != 0) {
            perror("ERROR: fail to join thread ...\n");
        }
    }
    pthread_barrier_destroy(&barrier);
}

void *knnPthreads(void *args) {
    struct Matrix_t *localArgs = (struct Matrix_t*)args;
    int threadId = localArgs->threadId;
    float* record = localArgs->record;
    int* topK;
    pthread_barrier_t *barrier = localArgs->barrier;

    for(int j=0; j<N; j++) {
        if(j % NUM_THREADS == threadId) {
            for(int i=0; i<M; i++) {
                record_total[i*N+j] = fvec_L2sqr_sse(A+i*D, B+j*D, D-1);
                //record_total[i*N+j] = getEuclideanDistance(A, i, B, j, D);
            }
        }
    }
    pthread_barrier_wait(barrier);

    for(int i=0; i<M; i++) {
        if(i % NUM_THREADS == threadId) {
            memcpy(record, &record_total[i*N], N * sizeof(float));
            topK = findTopK(record, N, K);
            A[i*D+D-1] = labelVectorArray(topK, K, B, N, D);
        }
    }

    return NULL;
}
