#include <cstdio>
#include <time.h>
#include <hip/hip_runtime.h>

#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
  if (err != hipSuccess) {
    printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

/* A simple GPU kernel definition */
__global__ void kernel(int *d_a, int n_total)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_total)
    d_a[idx] = idx;
}

/* The main function */
int main(){
  // Problem size
  constexpr int n_total = 4194304; // pow(2, 22);

  // Device grid sizes
  constexpr int blocksize = 256;
  constexpr int gridsize = (n_total - 1 + blocksize) / blocksize;

  // Allocate host and device memory
  int *a, *d_a;
  const int bytes = n_total * sizeof(int);
  HIP_ERRCHK(hipHostMalloc((void**)&a, bytes)); // host pinned
  HIP_ERRCHK(hipMalloc((void**)&d_a, bytes));   // device pinned

  // Create events
  hipEvent_t start_kernel_event;
  hipEvent_t start_d2h_event;
  hipEvent_t stop_event;

  HIP_ERRCHK(hipEventCreate(&start_kernel_event));
  HIP_ERRCHK(hipEventCreate(&start_d2h_event));
  HIP_ERRCHK(hipEventCreate(&stop_event));

  // Create stream
  hipStream_t stream;
  HIP_ERRCHK(hipStreamCreate(&stream));

  // Start timed GPU kernel and device-to-host copy
  HIP_ERRCHK(hipEventRecord(start_kernel_event, stream));
  clock_t start_kernel_clock = clock();
  kernel<<<gridsize, blocksize, 0, stream>>>(d_a, n_total);

  HIP_ERRCHK(hipEventRecord(start_d2h_event, stream));
  clock_t start_d2h_clock = clock();
  HIP_ERRCHK(hipMemcpyAsync(a, d_a, bytes, hipMemcpyDeviceToHost, stream));

  HIP_ERRCHK(hipEventRecord(stop_event, stream));
  clock_t stop_clock = clock();
  HIP_ERRCHK(hipStreamSynchronize(stream));

  // Exctract elapsed timings from event recordings
  float kernel_time;
  float d2h_time;
  float total_time;

  HIP_ERRCHK(hipEventElapsedTime(&kernel_time, start_kernel_event, start_d2h_event));
  HIP_ERRCHK(hipEventElapsedTime(&d2h_time, start_d2h_event, stop_event));
  HIP_ERRCHK(hipEventElapsedTime(&total_time, start_kernel_event, stop_event));

  // Check that the results are right
  int error = 0;
  for(int i = 0; i < n_total; ++i){
    if(a[i] != i)
      error = 1;
  }

  // Print results
  if(error)
    printf("Results are incorrect!\n");
  else
    printf("Results are correct!\n");

  // Print event timings
  printf("Event timings:\n");
  printf("  %.3f ms - kernel\n", kernel_time);
  printf("  %.3f ms - device to host copy\n", d2h_time);
  printf("  %.3f ms - total time\n", total_time);

  // Print clock timings
  printf("clock_t timings:\n");
  printf("  %.3f ms - kernel\n", 1e3 * (double)(start_d2h_clock - start_kernel_clock) / CLOCKS_PER_SEC);
  printf("  %.3f ms - device to host copy\n", 1e3 * (double)(stop_clock - start_d2h_clock) / CLOCKS_PER_SEC);
  printf("  %.3f ms - total time\n", 1e3 * (double)(stop_clock - start_kernel_clock) / CLOCKS_PER_SEC);

  // Destroy Stream
  HIP_ERRCHK(hipStreamDestroy(stream));

  // Destroy events
  HIP_ERRCHK(hipEventDestroy(start_kernel_event));
  HIP_ERRCHK(hipEventDestroy(start_d2h_event));
  HIP_ERRCHK(hipEventDestroy(stop_event));

  // Deallocations
  HIP_ERRCHK(hipFree(d_a)); // Device
  HIP_ERRCHK(hipHostFree(a)); // Host
}
