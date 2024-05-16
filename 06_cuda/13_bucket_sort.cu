#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init(int *bucket) { bucket[threadIdx.x] = 0; }

__global__ void count(int *bucket, int *keys) { atomicAdd(&bucket[keys[threadIdx.x]], 1); }

__global__ void sort(int *bucket, int *d_keys, int *keys) {
  if(threadIdx.x < bucket[blockIdx.x]) {
    keys[threadIdx.x + d_keys[blockIdx.x]] = blockIdx.x;
  }
}

__global__ void scan(int *bucket, int *d_keys, int range) {
  d_keys[threadIdx.x + 1] = bucket[threadIdx.x];
  for (int i = 1; i < range; i *= 2) {
    __syncthreads();
    int t = d_keys[threadIdx.x];
    int d = threadIdx.x - i;
    if(d >= 0) {
      t += d_keys[d];
    }
    __syncthreads();
    d_keys[threadIdx.x] = t;
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *keys;
  cudaMallocManaged(&keys, n * sizeof(int));
  for (int i = 0; i < n; i++) {
    keys[i] = rand() % range;
    printf("%d ", keys[i]);
  }
  printf("\n");

  /*
    std::vector<int> bucket(range); 
  for (int i=0; i<range; i++) {
    bucket[i] = 0;
  }
  for (int i=0; i<n; i++) {
    bucket[key[i]]++;
  }
  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }
  */
  
  int *bucket, *d_keys;
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&d_keys, (range + 1)*sizeof(int));
  init<<<1, range>>>(bucket);
  cudaDeviceSynchronize();
  count<<<1, n>>>(bucket, keys);
  cudaDeviceSynchronize();
  scan<<<1, range>>>(bucket, d_keys, range);
  cudaDeviceSynchronize();
  sort<<<range, n>>>(bucket, d_keys, keys);
  cudaDeviceSynchronize();

  cudaFree(bucket);
  cudaFree(d_keys);
  for (int i = 0; i < n; i++) {
    printf("%d ", keys[i]);
  }
  printf("\n");
  cudaFree(keys);
}
