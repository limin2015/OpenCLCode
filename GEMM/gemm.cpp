
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#include <iostream>
#include <cstring>
#include <fstream>

#include <CL/cl.h>


using namespace std;

#include "util.h"

#define CHECK_GEMM 1

//#pragma comment (lib,"OpenCL.lib")

inline void CHECK_MALLOC(void* fp){
	if(fp == NULL){
		printf("host malloc failed\n");
		exit(1);
	}
}

#define CHECK_CL(err) _CHECK_CL(err, __LINE__)
inline void _CHECK_CL(cl_int err, int line){
	if(err != CL_SUCCESS){
		printf("cl error(code=%d) at line %d.\n", err, line);	
		exit(1);
	}
} 

void checkGEMMResult(TYPE* C_host, TYPE* C, int len){
	int count = 0;
	for(int i=0; i<len; i++){
		if(fabs(C_host[i] - C[i]) > 1e-6){
			printf("C_host[%d]=%lf, C[%d]=%lf\n", i, C_host[i], i, C[i]);
			count++;
		}
	}
	
	printf("GEMM has %d ERR\n", count);
}

static double dtime(){
	double tseconds = 0.0;
	struct timeval mytime;
	gettimeofday(&mytime,NULL);
	tseconds = (double)((mytime.tv_sec) + (mytime.tv_usec*1e-6));
return tseconds;
}

void host_gemm(TYPE* A, 
			   TYPE* B, 
			   TYPE* C, 
			   int M, 
			   int N, 
			   int K){

	TYPE temp = 0.f;
	
	for(int i=0; i<M; i++)
		for(int j=0; j<N; j++){
			temp = 0.;
			for(int k=0; k<K; k++){
				temp += A[i + k * M] * B[k + j * K];
			}
			C[i + j * M] = temp;
		}
}


//把文本文件读入一个 string 中
int convertToString(const char *filename, std::string& s)
{
    size_t size;
    char* str;
    std::fstream f(filename, (std::fstream::in | std::fstream::binary));
    if (f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);
        str = new char[size + 1];
        if (!str)
        {
            f.close();
            return 1;
        }
        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
        s = str;
        delete[] str;
        return 0;
    }
    printf("Error: Failed to open file %s\n", filename);
    return 1;
}

int main(int argc, char* argv[])
{
	if(argc < 5){
		printf("Usage: N, M, P, iters.\n");
		exit(0);
	} 
    
	const int Ndim = atoi(argv[1]);   
    const int Mdim = atoi(argv[2]);   
    const int Pdim = atoi(argv[3]);   
    const int iters= atoi(argv[4]);   
	
	int m = Ndim;
	int n = Mdim;
	int k = Pdim;
	int bs = BS;
	printf("m=%d, n=%d, k=%d, blocksize=%d, iters=%d.\n", Ndim, Mdim, Pdim, bs, iters);
		
	// Host data
    cl_uint status;
    cl_platform_id platform;

    status = clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, 
				CL_DEVICE_TYPE_GPU,
        		1,
        		&device,
        		NULL);

    cl_context context = clCreateContext(NULL,
        								1,
        								&device,
        								NULL, NULL, NULL);

    cl_command_queue commandQueue = clCreateCommandQueue(context,
        												device,
        												CL_QUEUE_PROFILING_ENABLE, 
														NULL);

    if (commandQueue == NULL) 
            perror("Failed to create commandQueue for device 0.");

    cl_device_fp_config DeviceDouble;
	if(DOUBLE){
    	int Doubleflag; //1：支持双精度；0：不支持双精度
    	Doubleflag=1; //
    	status = clGetDeviceInfo(device, 
								CL_DEVICE_DOUBLE_FP_CONFIG,
    							sizeof(cl_device_fp_config), 
								&DeviceDouble, 
								NULL);
    	if(0 == DeviceDouble){
     		printf("Don't support double\n");
			exit(1);
		}
    	else printf("Support double\n");
  	}
 
	
	size_t  maxWorkItemPerGroup;
   	clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(maxWorkItemPerGroup), &maxWorkItemPerGroup, NULL);
   	printf("maxWorkItemPerGroup: %zd\n", maxWorkItemPerGroup);
 
    //const int TS = 16;
    int szA = Ndim * Pdim;
    int szB = Pdim * Mdim;
    int szC = Ndim * Mdim;

    TYPE *A;
    TYPE *B;
    TYPE *C;
    TYPE *C_host;

    A = (TYPE *)malloc(szA * sizeof(TYPE));
    B = (TYPE *)malloc(szB * sizeof(TYPE));
    C = (TYPE *)malloc(szC * sizeof(TYPE));
    C_host = (TYPE *)malloc(szC * sizeof(TYPE));

    int i, j;
	srand( (unsigned)time( NULL ) );
    for (i = 0; i < szA; i++)
   		//A[i] = rand()%RAND;
        A[i] = 1.0;
    for (i = 0; i < szB; i++)
   		//B[i] = rand()%RAND;
        B[i] = 1.0;

    cl_mem memObjects[3] = { 0, 0, 0 };
    memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY |  CL_MEM_COPY_HOST_PTR,
        sizeof(TYPE)* szA, A, NULL);
    memObjects[1] = clCreateBuffer(context, CL_MEM_READ_ONLY |  CL_MEM_COPY_HOST_PTR,
        sizeof(TYPE)* szB, B, NULL);
    memObjects[2] = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(TYPE)* szC, C, NULL);
    if (memObjects[0] == NULL || memObjects[1] == NULL ||memObjects[2] == NULL) 
        perror("Error in clCreateBuffer.\n");

    const char * filename = "simply_gemm.cl";
    //const char * filename = "optimized_gemm.cl";
    std::string sourceStr;
    status = convertToString(filename, sourceStr);

    const char * source = sourceStr.c_str();
    size_t sourceSize[] = { strlen(source) };

    cl_program program = clCreateProgramWithSource(
        context,
        1,
        &source,
        sourceSize,
        NULL);

	if(DOUBLE){
    	if(DeviceDouble)
      		status = clBuildProgram(program, 1, &device, "-D FP_64", NULL, NULL);
    }
	else{
    	status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	}

    if (status)
        cout << status << "  !!!!!!!!" <<endl;
    if (status != 0)
    {
        printf("clBuild failed:%d\n", status);
        char tbuf[0x10000];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0x10000, tbuf,
            NULL);
        printf("\n%s\n", tbuf);
        //return −1;
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_mul", NULL);


    //cl_int clnum = NWITEMS;
    status = clSetKernelArg(kernel, 0, sizeof(int), &Ndim);
    status = clSetKernelArg(kernel, 1, sizeof(int), &Mdim);
    status = clSetKernelArg(kernel, 2, sizeof(int), &Pdim);
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjects[0]);
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjects[1]);
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObjects[2]);
    if (status)
        cout << "Error when set kernel args" << endl;

    size_t global[2];
    cl_event prof_event;
    cl_ulong ev_start_time = (cl_ulong)0;
    cl_ulong ev_end_time = (cl_ulong)0;
    double cl_time;
    
	global[0] = (size_t)Ndim;
    global[1] = (size_t)Mdim;
    size_t localWorkSize[2] ;
    localWorkSize[0]= BS;
    localWorkSize[1]= BS;

//for(int i=0; i<iters; i++){
    status = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
             global, localWorkSize, 0, NULL, &prof_event);
    if (status)
        cout << "Error when execute kernel!" << endl;
//}	
	clFinish(commandQueue);
	status = clWaitForEvents(1, &prof_event);

    status = clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_QUEUED,
        sizeof(cl_ulong),&ev_start_time,NULL);
    status = clGetEventProfilingInfo(prof_event,CL_PROFILING_COMMAND_END,
        sizeof(cl_ulong),&ev_end_time,NULL);
    if (status) 
        perror("Error when acquire execution time!\n");
    cl_time = (double)(ev_end_time - ev_start_time) / 1e9;

    status = clEnqueueReadBuffer(commandQueue, memObjects[2], CL_TRUE, 0,
            sizeof(TYPE)* szC, C,0, NULL, NULL);
    if (status) 
        perror("Error when read back data.\n");
	
	//run host_gemm
	double t1 = dtime();
	host_gemm(A, B, C_host, m, n, k);
	double t2 = dtime();
	
	double cpu_time = t2 - t1;

    // Verify the output
	{
		//TODO
	#ifdef CHECK_GEMM
		checkGEMMResult(C_host, C, m*n);
	#endif
	}

	if(DOUBLE)
		printf("DOUBLE GEMM TEST:\n");
	else
		printf("FLOAT GEMM TEST:\n");
	

	printf("cl_time = %lf s.\n", cl_time);
	printf("cpu-time = %lf s.\n", cpu_time);	
	printf("speedup = %lf\n", cpu_time/(cl_time));


    if (A)
        free(A);
    if (B)
        free(B);
    if (C)
        free(C);
    if (C_host)
        free(C_host);

    clReleaseMemObject(memObjects[2]);
    clReleaseMemObject(memObjects[1]);
    clReleaseMemObject(memObjects[0]);
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);

    return 0;
}
