
void swapArray(int *arr, int a, int b) {
    int tmp = arr[b];
    arr[b] = arr[a];
    arr[a] = tmp;
}

int partition(int *topK, float *distanceArr, int left, int right) {
    int target = topK[right];
    int id = left - 1;
    for(int i=left; i<right; i++) {
        if(distanceArr[topK[i]] <= distanceArr[target]) {
            id++;
            swapArray(topK, id, i);
        }
    }
    swapArray(topK, id+1, right);
    return id+1;
}

int randomPartition(int* topK, float* distanceArr, int left, int right) {
    int target = rand() % (right - left + 1) + left;
    swapArray(topK, target, right);
    return partition(topK, distanceArr, left, right);
}


void quickSelect(int* topK, float* distanceArr, int K, int left, int right) {
    int pivot = randomPartition(topK, distanceArr, left, right);
    //printf("pivot %d", pivot);
    if(pivot == K) {
        return;
    } else if(pivot < K) {
        quickSelect(topK, distanceArr, K, pivot+1, right);
    } else {
        quickSelect(topK, distanceArr, K, left, pivot-1);
    }
}

int* findTopK(float* distanceArr, int N, int K) {
    int *topK = (int*)malloc(N*sizeof(int));
    for(int i=0; i<N; i++) {
        topK[i] = i;
    }
    srand(time(0));
    quickSelect(topK, distanceArr, K, 0, N-1);
    return topK;
}