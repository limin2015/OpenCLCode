#include "util.h"

__kernel void matrix_mul(
		const int Ndim,
    const int Mdim,
    const int Pdim,
    __global const TYPE* A, 
    __global const TYPE* B, 
    __global TYPE* C)
{
	//A: Pdim * Ndim
	//B: Mdim * Pdim
	const int row = get_local_id(0);
	const int col = get_local_id(1);
	const int globalRow = get_group_id(0) * get_local_size(0) + row;
	const int globalCol = get_group_id(1) * get_local_size(1) + col;
	//local memory on chip
	__local TYPE Asub[BS][BS];
	__local TYPE Bsub[BS][BS];
	TYPE acc = 0.0f;
	
	const int numTiles = Pdim / BS;
	for (int t=0; t<numTiles; t++) {
      // Load one tile of A and B into local memory
      const int tiledRow = BS*t + row;
      const int tiledCol = BS*t + col;
      Asub[col][row] = A[tiledCol*Ndim + globalRow];
      Bsub[col][row] = B[globalCol*Pdim + tiledRow];
      
      //printf("t=%d\n", t);
      //printf("row=%d, col=%d, tiledRow=%d, tiledCol=%d,  globalRow=%d, globalCol=%d\n", row, col, tiledRow, tiledCol, globalRow, globalCol);
      barrier(CLK_LOCAL_MEM_FENCE);

      // Perform the computation for a single tile
      for (int k=0; k<BS; k++) {
          acc += Asub[k][row] * Bsub[col][k];
      }
      
      barrier(CLK_LOCAL_MEM_FENCE);
    }
	
	C[globalCol * Mdim + globalRow] = acc;

}

