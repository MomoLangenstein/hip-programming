#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

/* HIP error handling macro */
#define HIP_ERRCHK(err) (hip_errchk(err, __FILE__, __LINE__ ))
static inline void hip_errchk(hipError_t err, const char *file, int line) {
  if (err != hipSuccess) {
    printf("\n\n%s in %s at line %d\n", hipGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}

__global__ void saxpy_(const int n, const float a, const float* const x, float* const y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

int main(void)
{
    int i;
    const int n = 10000;
    float a = 3.4;
    float x[n], y[n], y_ref[n];
    float *x_, *y_;

    // initialise data and calculate reference values on CPU
    for (i=0; i < n; i++) {
        x[i] = sin(i) * 2.3;
        y[i] = cos(i) * 1.1;
        y_ref[i] = a * x[i] + y[i];
    }

    HIP_ERRCHK(hipMalloc(&x_, sizeof(float) * n));
    HIP_ERRCHK(hipMalloc(&y_, sizeof(float) * n));

    HIP_ERRCHK(hipMemcpy(x_, x, sizeof(float) * n, hipMemcpyHostToDevice));
    HIP_ERRCHK(hipMemcpy(y_, y, sizeof(float) * n, hipMemcpyHostToDevice));

    dim3 blocks(32);
    dim3 threads(256);

    saxpy_<<<blocks, threads>>>(n, a, x_, y_);

    HIP_ERRCHK(hipMemcpy(y, y_, sizeof(float) * n, hipMemcpyDeviceToHost));

    HIP_ERRCHK(hipFree(x_));
    HIP_ERRCHK(hipFree(y_));

    // confirm that results are correct
    float error = 0.0;
    float tolerance = 1e-6;
    float diff;
    for (i=0; i < n; i++) {
        diff = abs(y_ref[i] - y[i]);
        if (diff > tolerance)
            error += diff;
    }
    printf("total error: %f\n", error);
    printf("  reference: %f at (42)\n", y_ref[42]);
    printf("     result: %f at (42)\n", y[42]);

    return 0;
}
