#include <stdio.h>
#include <math.h>
#include <hip/hip_runtime.h>

/* HIP error handling macro */
#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
  if (err != hipSuccess) {
    printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

__global__ void copy2d_(const int n, const int m, const double* const x, double* const y) {
    int column = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int tid = row * n + column;

    if (column < n && row < m) {
        y[tid] = x[tid];
    }
}

int main(void)
{
    const int n = 600;
    const int m = 400;
    const int size = n * m;
    double x[size], y[size], y_ref[size];
    double *x_, *y_;

    // initialise data
    for (int i=0; i < size; i++) {
        x[i] = (double) i / 1000.0;
        y[i] = 0.0;
    }
    // copy reference values (C ordered)
    for (int j=0; j < m; j++) {
      for (int i=0; i < n; i++) {
        y_ref[j * n + i] = x[j * n + i];
      }
    }

    HIP_ERRCHK(hipMalloc(&x_, sizeof(double) * size));
    HIP_ERRCHK(hipMalloc(&y_, sizeof(double) * size));

    HIP_ERRCHK(hipMemcpy(x_, x, sizeof(double) * size, hipMemcpyHostToDevice));

    dim3 blocks((n + 63) / 64, m);
    dim3 threads(64);

    copy2d_<<<blocks, threads>>>(n, m, x_, y_);

    HIP_ERRCHK(hipMemcpy(y, y_, sizeof(double) * size, hipMemcpyDeviceToHost));

    HIP_ERRCHK(hipFree(x_));
    HIP_ERRCHK(hipFree(y_));

    // confirm that results are correct
    double error = 0.0;
    for (int i=0; i < size; i++) {
        error += abs(y_ref[i] - y[i]);
    }
    printf("total error: %f\n", error);
    printf("  reference: %f at (42,42)\n", y_ref[42 * m + 42]);
    printf("     result: %f at (42,42)\n", y[42 * m + 42]);

    return 0;
}
