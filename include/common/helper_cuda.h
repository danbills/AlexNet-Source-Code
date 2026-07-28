#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#define cutilCheckMsg(msg) checkCudaErrors(cudaGetLastError())
#define cutilSafeCall(val) checkCudaErrors(val)

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

template<typename T>

void check(T err, const char* const func, const char* const file, const int line) {
    if (err) {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\"\n", file, line, static_cast<unsigned int>(err), func);
        exit(EXIT_FAILURE);
    }
}

#endif
