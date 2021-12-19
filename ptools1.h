#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#include "quickSelect.h"
#include "distance.h"

#define MATRIX_DIMENSION 256
#define TOP_K 5

float* generateMatrix(int M, int N) {
    float *arr = (float*)malloc((M*N)*sizeof(float));
    srand(time(NULL));
    for(int i=0; i<M; i++) {
        for(int j=0; j<N-1; j++) {
            arr[i*N+j] = ((float)rand()/(float)(RAND_MAX)) * 10;
        }
        arr[i*N+N-1] = (float)(rand() % 2);
    }
    return arr;
}

void printMatrix(float *matrix, int M, int N) {
    for(int i=0; i<M; i++) {
        for(int j=0; j<N; j++) {
            printf("%f ", matrix[i*N+j]);
        }
        printf("\n");
    }
}

void validateMatrix(float *A, float *B, int M, int N) {
    float epsilon = 0.001;
    for(int i=0; i<M; i++) {
        for(int j=0; j<N; j++) {
            if(fabs(A[i*N+j]-B[i*N+j]) > epsilon) {
                printf("error, row: %d, col: %d, A: %f B: %f\n", i, j, A[i*N+j], B[i*N+j]);
                return;
            }
        }
    }
    printf("Validation successful!\n");
}

float getEuclideanDistance(float *A, int i, float *B, int j, int D) {
    float res = 0.0;
    for(int k=0; k<D-1; k++) {
        res += pow(A[i*D+k]-B[j*D+k], 2);
    }
    return sqrtf(res);
}

void printLabel(int label, int row) {
    if(label == 0) {
        printf("Vector %d can be labeled as dog\n", row);
        return;
    }
    printf("Vector %d can be labeled as cat\n", row);
}

int labelVectorArray(int *topK, int K, float* B, int N, int D) {
    int cnt0 = 0;
    int cnt1 = 0;
    for(int i=0; i<K; i++) {
        int row = topK[i];
        // printf("Current row: %.2f\n", B[row*D+D-1]);
        if(B[row*D+D-1] < 0.5f) {
            cnt0++;
        } else {
            cnt1++;
        }
    }
    return cnt0 > cnt1 ? 0 : 1;
}

void validateLables(float *A, float *B, int M, int D) {
    for(int i=0; i<M; i++) {
        if(A[i*D+D-1] != B[i*D+D-1]) {
            printf("ERROR: validation fails at query %d\n", i);
            return;
        }
    }
    printf("Validation Successful!\n");
}

void serialKNN1(float *A, float *B, int M, int N, int D, int K) {
    float* record = (float*)malloc(N*sizeof(float));
    int* topK;

    for(int i=0; i<M; i++) {
        for(int j=0; j<N; j++) {
            //record[j] = getEuclideanDistance(A, i, B, j, D);
            record[j] = fvec_L2sqr_sse(A+i*D, B+j*D, D-1);
        }
        topK = findTopK(record, N, K);
        A[i*D+D-1] = labelVectorArray(topK, K, B, N, D);
    }
}

void labelMatrices(float *A, float *record, float *B, int M, int N, int D, int K) {
    float *row = (float*)malloc(N*sizeof(float));
    int* topK;
    for(int i=0; i<M; i++) {
        memcpy(row, &record[i*N], N * sizeof(float));
        topK = findTopK(row, N, K);
        A[i*D+D-1] = labelVectorArray(topK, K, B, N, D);
    }
}
