#include <cstdio>
#include <string>
#include <time.h>
#include <hip/hip_runtime.h>

/* Blocksize divisible by the warp size */
#define BLOCKSIZE 64

/* GPU kernel definition */
__global__ void hipKernel(int* const A, const int nx, const int ny)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nx * ny)
  {
    const int i = idx % nx;
    const int j = idx / nx;
    A[j * nx + i] += idx;
  }
}

/* Auxiliary function to check the results */
void checkResults(int* const A, const int nx, const int ny, const std::string strategy, const double timing)
{
  // Check that the results are correct
  int errored = 0;
  for(unsigned int i = 0; i < nx * ny; i++)
    if(A[i] != i)
      errored = 1;

  // Indicate if the results are correct
  if(errored)
    printf("The results are incorrect!/n");
  else
    printf("The results are OK! (%.3fs - %s)\n", timing, strategy.c_str());
}

/* Run using explicit memory management */
void explicitMem(int nSteps, int nx, int ny)
{
  // Determine grid size
  const int gridsize = (nx * ny - 1 + BLOCKSIZE) / BLOCKSIZE;

  int *A, *d_A;
  size_t size = nx * ny * sizeof(int);

  A = (int*) malloc(size);
  hipMalloc(&d_A, size);

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent a common
     * workflow of a GPU accelerated program:
     * Accessing the array from host,
     * copying data from host to device,
     * and running a GPU kernel.
     */

    // Initialize array from host
    memset(A, 0, size);

    hipMemcpy(d_A, A, size, hipMemcpyHostToDevice);

    // Launch GPU kernel
    hipLaunchKernelGGL(hipKernel,
      gridsize, BLOCKSIZE, 0, 0,
      d_A, nx, ny);

    hipStreamSynchronize(0);
  }

  hipMemcpy(A, d_A, size, hipMemcpyDeviceToHost);

  // Check results and print timings
  clock_t tStop = clock();
  checkResults(A, nx, ny, "ExplicitMemCopy", (double)(tStop - tStart) / CLOCKS_PER_SEC);

  hipFree(d_A);
  free(A);
}

/* Run using explicit memory management and pinned host allocations */
void explicitMemPinned(int nSteps, int nx, int ny)
{
  // Determine grid size
  const int gridsize = (nx * ny - 1 + BLOCKSIZE) / BLOCKSIZE;

  int *A, *d_A;
  size_t size = nx * ny * sizeof(int);

  hipMallocHost((void**)&A, size);
  hipMalloc(&d_A, size);

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent a common
     * workflow of a GPU accelerated program:
     * Accessing the array from host,
     * copying data from host to device,
     * and running a GPU kernel.
     */

    // Initialize array from host
    memset(A, 0, size);

    hipMemcpy(d_A, A, size, hipMemcpyHostToDevice);

    // Launch GPU kernel
    hipLaunchKernelGGL(hipKernel,
      gridsize, BLOCKSIZE, 0, 0,
      d_A, nx, ny);

    hipStreamSynchronize(0);
  }

  hipMemcpy(A, d_A, size, hipMemcpyDeviceToHost);

  // Check results and print timings
  clock_t tStop = clock();
  checkResults(A, nx, ny, "ExplicitMemPinnedCopy", (double)(tStop - tStart) / CLOCKS_PER_SEC);

  hipFree(d_A);
  hipFreeHost(A);
}

/* Run using explicit memory management without recurring host/device memcopies */
void explicitMemNoCopy(int nSteps, int nx, int ny)
{
  // Determine grid size
  const int gridsize = (nx * ny - 1 + BLOCKSIZE) / BLOCKSIZE;

  int *A, *d_A;
  size_t size = nx * ny * sizeof(int);

  A = (int*) malloc(size);
  hipMalloc(&d_A, size);

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent an optimal
     * workflow of a GPU accelerated program where all oprations
     * are performed using device (ie, recurring memcopy is avoided):
     * Initializing array using device, and running a GPU kernel.
     */

    hipMemset(d_A, 0, size);

    // Launch GPU kernel
    hipLaunchKernelGGL(hipKernel,
      gridsize, BLOCKSIZE, 0, 0,
      d_A, nx, ny);
  }

  hipMemcpy(A, d_A, size, hipMemcpyDeviceToHost);

  // Check results and print timings
  clock_t tStop = clock();
  checkResults(A, nx, ny, "ExplicitMemNoCopy", (double)(tStop - tStart) / CLOCKS_PER_SEC);

  hipFree(d_A);
  free(A);
}

/* Run using Unified Memory */
void unifiedMem(int nSteps, int nx, int ny)
{
  // Determine grid size
  const int gridsize = (nx * ny - 1 + BLOCKSIZE) / BLOCKSIZE;

  int *A;
  size_t size = nx * ny * sizeof(int);

  hipMallocManaged(&A, size);

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent
     * a common workflow of a GPU accelerated program:
     * Accessing the array from host,
     * (data copy from host to device is handled automatically),
     * and running a GPU kernel.
     */

    // Initialize array from host
    memset(A, 0, size);

    hipLaunchKernelGGL(hipKernel,
      gridsize, BLOCKSIZE, 0, 0,
      A, nx, ny);

    hipStreamSynchronize(0);
  }

  // Check results and print timings
  clock_t tStop = clock();
  checkResults(A, nx, ny, "UnifiedMemNoPrefetch", (double)(tStop - tStart) / CLOCKS_PER_SEC);

  hipFree(A);
}

/* Run using Unified Memory and prefetching */
void unifiedMemPrefetch(int nSteps, int nx, int ny)
{
  // Determine grid size
  const int gridsize = (nx * ny - 1 + BLOCKSIZE) / BLOCKSIZE;

  int device;
  hipGetDevice(&device);
  int *A;
  size_t size = nx * ny * sizeof(int);

  hipMallocManaged(&A, size);

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent a common
     * workflow of a GPU accelerated program:
     * Accessing the array from host,
     * prefetching data from host to device,
     * and running a GPU kernel.
     */

    // Initialize array from host
    memset(A, 0, size);

    hipMemPrefetchAsync(A, size, device, 0);

    hipLaunchKernelGGL(hipKernel,
      gridsize, BLOCKSIZE, 0, 0,
      A, nx, ny);

    hipStreamSynchronize(0);
  }

  hipMemPrefetchAsync(A, size, hipCpuDeviceId, 0);

  hipStreamSynchronize(0);

  // Check results and print timings
  clock_t tStop = clock();
  checkResults(A, nx, ny, "UnifiedMemPrefetch", (double)(tStop - tStart) / CLOCKS_PER_SEC);

  hipFree(A);
}

/* Run using Unified Memory without recurring host/device memcopies */
void unifiedMemNoCopy(int nSteps, int nx, int ny)
{
  // Determine grid size
  const int gridsize = (nx * ny - 1 + BLOCKSIZE) / BLOCKSIZE;

  int device;
  hipGetDevice(&device);

  int *A;
  size_t size = nx * ny * sizeof(int);

  hipMallocManaged(&A, size);

  // Start timer and begin stepping loop
  clock_t tStart = clock();
  for(unsigned int i = 0; i < nSteps; i++)
  {
    /* The order of calls inside this loop represent an optimal
     * workflow of a GPU accelerated program where all oprations
     * are performed using device (ie, recurring memcopy is avoided):
     * Initializing array using device, and running a GPU kernel.
     */

    hipMemset(A, 0, size);

    // Launch GPU kernel
    hipLaunchKernelGGL(hipKernel,
      gridsize, BLOCKSIZE, 0, 0,
      A, nx, ny);
  }

  hipMemPrefetchAsync(A, size, hipCpuDeviceId, 0);

  hipStreamSynchronize(0);

  // Check results and print timings
  clock_t tStop = clock();
  checkResults(A, nx, ny, "UnifiedMemNoCopy", (double)(tStop - tStart) / CLOCKS_PER_SEC);

  hipFree(A);
}

/* The main function */
int main(int argc, char* argv[])
{
  // Set the number of steps and 2D grid dimensions
  int nSteps = 100, nx = 8000, ny = 2000;

  // Run with different memory management strategies
  explicitMem(nSteps, nx, ny);
  explicitMemPinned(nSteps, nx, ny);
  explicitMemNoCopy(nSteps, nx, ny);
  unifiedMem(nSteps, nx, ny);
  unifiedMemPrefetch(nSteps, nx, ny);
  unifiedMemNoCopy(nSteps, nx, ny);
}
