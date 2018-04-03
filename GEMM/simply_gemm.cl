#include "util.h"

__kernel void matrix_mul(
    const int Ndim,
    const int Mdim,
    const int Pdim,
    __global const TYPE* A, 
    __global const TYPE* B, 
    __global TYPE* C)
{
    //A: Ndim * Pdim
    //B: Pdim * Mdim
    int i = get_global_id(0);
    int j = get_global_id(1);

    int k;
    TYPE tmp;

    if ((i < Ndim) && (j < Mdim)) {
        tmp = 0.0;
        for (k = 0; k < Pdim; k++)
            tmp += A[i*Pdim + k] * B[k*Mdim + j];
        C[i*Mdim + j] = tmp;
    }
}
