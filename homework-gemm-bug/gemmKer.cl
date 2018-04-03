__kernel void gemm_cl_v1(__global const TYPE* A, 
					     __global const TYPE* B, 
						 __global TYPE* C, 
					     const int M, 
					     const int N, 
					     const int K){

	int i = get_global_id(0); 
	int j = get_global_id(1); 

	int k;
	TYPE temp;

	//if(i == 0 && j == 0)
		//printf("A[0]=%lf, B[0]=%lf\n", A[0], B[0]);
	
	if(i < M && j < N){
		temp = 0.;
		for(k=0; k<K; k++){
			temp += A[i + k * M] * B[k + j * K];
		}
		C[i + j * M] = temp;
	}
}
