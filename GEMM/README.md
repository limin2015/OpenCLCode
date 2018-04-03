
/***********************************************************

Author: Min Li(iscas)

Time: 2018.4.1

Func:GEMM implementaiton using OpenCL.

    C[N][M] = A[N][P] * B[P][M].//col-priority.

Usage:

	Compile:

		sh build.sh

	Run:	

		sh run.sh 

		Note: in "run.sh" file, users can set values for: N, M, P, iters.
	
	To test	DOUBLE type gemm:

		in "util.h", you can set double type gemm by uncomment:	

			//#define TYPE double
			//#define DOUBLE 1
	
	To test	FLOAT type gemm:

		in "util.h", you can set double type gemm by uncomment:	

			//#define TYPE float 
			//#define DOUBLE 0

	Note: In default, opencl gemm's correctness is comparied with host gemm.
		  If you don't want to compare, comment	"#define CHECK_GEMM 1" in "gemm.cpp"

	**分块大小可以随便设置吗?**

		本文设置的默认的BS(分块大小)值为16.

		可以使用如下的方式查询：localworksize的参数(本文是16*16)

			size_t      maxWorkItemPerGroup;
   			clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(maxWorkItemPerGroup), &maxWorkItemPerGroup, NULL);
   			printf("maxWorkItemPerGroup: %zd\n", maxWorkItemPerGroup);

		查询出来后，只要：localWorkSize[0]*localWorkSize[1] <= CL_DEVICE_MAX_WORK_GROUP_SIZE即可。
		
***********************************************************/

