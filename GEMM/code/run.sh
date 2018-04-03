## m, n, k, iters.
FILE=resFloat-simple.txt
##FILE=resDouble.txt
./gemm 16 16 16  1 |tee $FILE 
./gemm 64 64 64 1 |tee >>$FILE 
./gemm 256 256 256 1 |tee >>$FILE 
./gemm 512 512 512 1 |tee >>$FILE 
./gemm 1024 1024 1024 1 |tee >>$FILE 
./gemm 2048 2048 2048 1 |tee >>$FILE 
./gemm 4096 4096 4096 1 |tee >>$FILE 
##./gemm 8192 8192 8192 1 |tee >>$FILE 
