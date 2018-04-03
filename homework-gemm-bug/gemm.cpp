#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
// OpenCL includes
#include <CL/cl.h>
//user define includes.
#include "util.h"


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
	double tuseconds = 0.0;
	struct timeval mytime;
	gettimeofday(&mytime,NULL);
	tuseconds = (double)((mytime.tv_sec*1.0e6) + (mytime.tv_usec));
return tuseconds;
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


void host_judge_double(){

}


void cl_preprocess(cl_context* context, 
				 	cl_uint* numDevices, 
					cl_device_id** devices, 
					cl_command_queue* cmdQueue, 
					cl_mem* bufferA, 
					cl_mem* bufferB, 
					cl_mem* bufferC,
					TYPE* A, 
					TYPE* B, 
					TYPE* C,
					int aSize,
					int bSize,
					int cSize){
 
    cl_int status;  
    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;
    
	status = clGetPlatformIDs(0, NULL, &numPlatforms);        CHECK_CL(status);
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
    status = clGetPlatformIDs(numPlatforms, platforms, NULL); CHECK_CL(status);
    
    status = clGetDeviceIDs(
        platforms[0], 
        CL_DEVICE_TYPE_ALL, 
        0, 
        NULL, 
        numDevices);
	CHECK_CL(status);
    
	(*devices) = (cl_device_id*)malloc((*numDevices)*sizeof(cl_device_id));
	CHECK_MALLOC((*devices));

    status = clGetDeviceIDs(
        platforms[0], 
        CL_DEVICE_TYPE_ALL,        
        *numDevices, 
        *devices, 
        NULL);
	CHECK_CL(status);

	//STEP 3: create context
    *context = clCreateContext(
        NULL, 
        *numDevices, 
        *devices, 
        NULL, 
        NULL, 
        &status);
	CHECK_CL(status);

    // STEP 4: Create a command queue
    (*cmdQueue) = clCreateCommandQueue(
        *context, 
        *devices[0], 
        /*0,*/
		CL_QUEUE_PROFILING_ENABLE, 
        &status);
	CHECK_CL(status);

    // STEP 5: Create device buffers
    *bufferA = clCreateBuffer(
        *context, 
        CL_MEM_READ_ONLY,                         
        aSize, 
        NULL, 
        &status);
	CHECK_CL(status);

    *bufferB = clCreateBuffer(
        *context, 
        CL_MEM_READ_ONLY,                         
        bSize, 
        NULL, 
        &status);
	CHECK_CL(status);

    *bufferC = clCreateBuffer(
        *context, 
        CL_MEM_WRITE_ONLY,                 
        cSize, 
        NULL, 
        &status);
	CHECK_CL(status);
   
	// transfer data 
    status = clEnqueueWriteBuffer(
        *cmdQueue, 
        *bufferA, 
        CL_FALSE, 
        0, 
        aSize,                         
        A, 
        0, 
        NULL, 
        NULL);
	CHECK_CL(status);
    
    status = clEnqueueWriteBuffer(
        *cmdQueue, 
        *bufferB, 
        CL_FALSE, 
        0, 
        bSize,                                  
        B, 
        0, 
        NULL, 
        NULL);
	CHECK_CL(status);

	free(platforms);
}


void cl_postprocess(cl_device_id** devices){

	free(*devices);
}

int main(int argc, 
		char* argv[]){
    // This code executes on the OpenCL host
   
	if(argc < 5){
		printf("Usage: m, n, k, iters. \n");
		exit(0);
	} 
    // Host data
    TYPE *A = NULL;  // Input array
    TYPE *B = NULL;  // Input array
    TYPE *C = NULL;  // Output array
    TYPE *C_host = NULL;  // Output array
	 
    const int m = atoi(argv[1]);   
    const int n = atoi(argv[2]);   
    const int k = atoi(argv[3]);   
    const int iters= atoi(argv[4]);   
	printf("m=%d, n=%d, k=%d, iters=%d.\n", m, n, k, iters);
		
    
	int aSize = m * k * sizeof(TYPE);
	int bSize = k * n * sizeof(TYPE);
	int cSize = m * n * sizeof(TYPE);
	
    A = (TYPE* )malloc(aSize); 		CHECK_MALLOC(A);
    B = (TYPE* )malloc(bSize); 		CHECK_MALLOC(B);
    C = (TYPE* )malloc(cSize); 		CHECK_MALLOC(C);
    C_host = (TYPE* )malloc(cSize); CHECK_MALLOC(C_host);

	srand( (unsigned)time( NULL ) );
    // Initialize the input data
    for(int i = 0; i < m*k; i++) {
   		//A[i] = rand()%RAND;
	    A[i] = 1;
	}
    for(int i = 0; i < m*k; i++) {
   		//B[i] = rand()%RAND;
        B[i] = 1;
	}


    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;
    
    cl_context context = NULL;
    cl_command_queue queue;
    
	cl_mem bufferA;  // Input array on the device
    cl_mem bufferB;  // Input array on the device
    cl_mem bufferC;  // Output array on the device
	
	cl_preprocess(&context, 
				&numDevices, 
				&devices, 
				&queue, 
				&bufferA, 
				&bufferB, 
				&bufferC, 
				A, 
				B, 
				C,
				aSize,
				bSize,
				cSize);

	/***********read kernel source code***************/
	FILE* fp;
	long filelen;
	long readlen;
	char* kernel_src;

	fp = fopen("gemmKer.cl", "r");	
	fseek(fp, 0L, SEEK_END);
	filelen = ftell(fp);
	rewind(fp);
   
	kernel_src = (char* )malloc(sizeof(char)* (filelen+1)); 
	CHECK_MALLOC(kernel_src);
	
	readlen = fread(kernel_src, 1, filelen, fp); 
	if(readlen != filelen){
		printf("error reading gemmKern.cl\n");
		exit(1);
	}
	kernel_src[filelen] = '\0';
	/***********end of read kernel source code**********/

	//printf("src=%s\n", kernel_src);//read ok!!

    cl_int err;  
	// Create a program using clCreateProgramWithSource()
    cl_program program = clCreateProgramWithSource(
        context, 
        1, 
        (const char**)&kernel_src,                                 
        NULL, 
        &err);
	CHECK_CL(err);
    

	printf("numDevices=%d\n", numDevices);
	// Build (compile) the program for the devices with
    // clBuildProgram()
    err= clBuildProgram(
        program, 
        numDevices, 
        devices, 
        NULL, 
        NULL, 
        NULL);
	//CHECK_CL(err);
	if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[8 * 1024];
         
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, 0, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }


    cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "gemm_cl_v1", &err);
	CHECK_CL(err);
	
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&A); CHECK_CL(err);
	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&B); CHECK_CL(err);
	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&C); CHECK_CL(err);
	err = clSetKernelArg(kernel, 3, sizeof(int), (void*)&m);    CHECK_CL(err);
	err = clSetKernelArg(kernel, 4, sizeof(int), (void*)&n);    CHECK_CL(err);
	err = clSetKernelArg(kernel, 5, sizeof(int), (void*)&k);    CHECK_CL(err);

	const int wg_dim = 16;
	const size_t global[2] = {m, n};
	const size_t local[2] = {wg_dim, wg_dim};

	cl_event prof_event;
	cl_ulong ev_start_time = (cl_ulong)0;  
	cl_ulong ev_end_time = (cl_ulong)0;  
	double cl_time;  

	err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &prof_event);
	CHECK_CL(err);	
	
	clFinish(queue);
	err = clWaitForEvents(1, &prof_event);
	CHECK_CL(err);	
	//get start time and end time: unit(ns)   
	err = clGetEventProfilingInfo(prof_event,  
								  CL_PROFILING_COMMAND_QUEUED,  
								  sizeof(cl_ulong),  
								  &ev_start_time,  
								  NULL);  
	CHECK_CL(err);

   err = clGetEventProfilingInfo(prof_event,  
                                  CL_PROFILING_COMMAND_END,  
                                  sizeof(cl_ulong),  
                                  &ev_end_time,  
                                  NULL);  
	CHECK_CL(err);

	cl_time = (double)(ev_end_time - ev_start_time);  
	
 
	err = clEnqueueReadBuffer(
        queue, 
        bufferC, 
        CL_TRUE, 
        0, 
        cSize,//CHECK 
        C, 
        0, 
        NULL, 
        NULL);
	CHECK_CL(err);

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

	printf("cl_time = %lf us.\n", cl_time*1e-3);
	printf("cpu-time = %lf us.\n", cpu_time);	
	printf("speedup = %lf\n", cpu_time/(cl_time*1e-3));
	
	cl_postprocess(&devices);
	// Free OpenCL resources
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    
	clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
   
	clReleaseKernel(kernel);
    clReleaseProgram(program);

	free(kernel_src);

	free(A);
	free(B);
	free(C);
	free(C_host);

return 0;
}



#if 0
cl_device_fp_config DeviceDouble;
int DoubleFlag = 1;//1 represent support double;

//check CL_DEVICE_DOUBLE_FP_CONFIG
err = clGetDeviceInfo(device, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config), &DeviceDouble, NULL);

if(0 == DeviceDouble)
	Doubleflag = 0;

if(1 == DeviceDouble)
	clBuildProgram(program, 1, &device, "-D FP_64", NULL);
else
	clBuildProgram(program, 1, &device, NULL, NULL);

#endif
