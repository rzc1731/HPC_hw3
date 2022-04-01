#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  // TODO: implement multi-threaded OpenMP scan
  if (n == 0) return;

  int p = 10;

  long chunksize = ceil(n/(long)p);
  long constants[p];
  constants[0] = 0;

  #pragma omp parallel num_threads(p)
  {
    int tid = omp_get_thread_num();
    long start = tid * chunksize;
    long end = std::min(start + chunksize, n);
    if (tid == 0) {
      prefix_sum[0] = 0;
    } else {
      prefix_sum[start] = A[start - 1];
    }
    for (long i=start+1; i<end; i++){
      prefix_sum[i] = prefix_sum[i-1] + A[i-1];
    }

    #pragma omp barrier

    #pragma omp single
    {
      long const_index;
      for (int i=1; i<p; i++){
          const_index = std::min(i*chunksize, n)-1;
          constants[i] = constants[i-1] + prefix_sum[const_index];
      }
    }

    #pragma omp barrier

    for (long i=start; i<end; i++){
        prefix_sum[i] += constants[tid];
    }
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
