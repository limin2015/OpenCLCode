// This program implements a vector addition using OpenCL

// System includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

// OpenCL includes
#include <CL/cl.h>

static double dtime(){
	double tuseconds = 0.0;
	struct timeval mytime;
	gettimeofday(&mytime,NULL);
	tuseconds = (double)((mytime.tv_sec*1.0e6) + (mytime.tv_usec));
return tuseconds;
}
	
double hostVectorAdd(int* C, int* A, int* B, int len){
	double t1, t2;
	t1 = dtime();
	for(int i=0; i<len; i++){
		C[i] = A[i] + B[i];
	}
	t2 = dtime();
	return t2 - t1;
}	

// OpenCL kernel to perform an element-wise 
// add of two arrays                        
const char* programSource =
"__kernel                        \n"
"void vecadd(__global int *A,    \n"
"            __global int *B,    \n"
"            __global int *C)    \n"
"{                               \n"
"   int idx = get_global_id(0);  \n"
"   C[idx] = A[idx] + B[idx];    \n"
" }                              \n"  
;

int main(int argc, char* argv[]) {
    // This code executes on the OpenCL host
   
	if(argc < 3){
		printf("Usage: numOfElements, iters. \n");
		exit(0);
	} 
    // Host data
    int *A = NULL;  // Input array
    int *B = NULL;  // Input array
    int *C = NULL;  // Output array
    int *C_host = NULL;  // Output array
    
    // Elements in each array
    //const int elements = 2048;   
    const long long elements = atol(argv[1]);   
    const int iters= atoi(argv[2]);   
   
	printf("nElements=%lld.\n", elements);
 
    // Compute the size of the data 
    size_t datasize = sizeof(int)*elements;

    // Allocate space for input/output data
    A = (int*)malloc(datasize);
    B = (int*)malloc(datasize);
    C = (int*)malloc(datasize);
    C_host = (int*)malloc(datasize);
    // Initialize the input data
    for(int i = 0; i < elements; i++) {
        A[i] = i;
        B[i] = i;
    }

    // Use this to check the output of each API call
    cl_int status;  
     
    //-----------------------------------------------------
    // STEP 1: Discover and initialize the platforms
    //-----------------------------------------------------
    
    cl_uint numPlatforms = 0;
    cl_platform_id *platforms = NULL;
    
    // Use clGetPlatformIDs() to retrieve the number of 
    // platforms
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
 
    // Allocate enough space for each platform
    platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));
 
    // Fill in platforms with clGetPlatformIDs()
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);

    //-----------------------------------------------------
    // STEP 2: Discover and initialize the devices
    //----------------------------------------------------- 
    
    cl_uint numDevices = 0;
    cl_device_id *devices = NULL;

    // Use clGetDeviceIDs() to retrieve the number of 
    // devices present
    status = clGetDeviceIDs(
        platforms[0], 
        CL_DEVICE_TYPE_ALL, 
        0, 
        NULL, 
        &numDevices);

    // Allocate enough space for each device
    devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));

    // Fill in devices with clGetDeviceIDs()
    status = clGetDeviceIDs(
        platforms[0], 
        CL_DEVICE_TYPE_ALL,        
        numDevices, 
        devices, 
        NULL);

    //-----------------------------------------------------
    // STEP 3: Create a context
    //----------------------------------------------------- 
    
    cl_context context = NULL;

    // Create a context using clCreateContext() and 
    // associate it with the devices
    context = clCreateContext(
        NULL, 
        numDevices, 
        devices, 
        NULL, 
        NULL, 
        &status);

    //-----------------------------------------------------
    // STEP 4: Create a command queue
    //----------------------------------------------------- 
    
    cl_command_queue cmdQueue;

    // Create a command queue using clCreateCommandQueue(),
    // and associate it with the device you want to execute 
    // on
    cmdQueue = clCreateCommandQueue(
        context, 
        devices[0], 
        0, 
        &status);

    //-----------------------------------------------------
    // STEP 5: Create device buffers
    //----------------------------------------------------- 
    
    cl_mem bufferA;  // Input array on the device
    cl_mem bufferB;  // Input array on the device
    cl_mem bufferC;  // Output array on the device

    // Use clCreateBuffer() to create a buffer object (d_A) 
    // that will contain the data from the host array A
    bufferA = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY,                         
        datasize, 
        NULL, 
        &status);

    // Use clCreateBuffer() to create a buffer object (d_B)
    // that will contain the data from the host array B
    bufferB = clCreateBuffer(
        context, 
        CL_MEM_READ_ONLY,                         
        datasize, 
        NULL, 
        &status);

    // Use clCreateBuffer() to create a buffer object (d_C) 
    // with enough space to hold the output data
    bufferC = clCreateBuffer(
        context, 
        CL_MEM_WRITE_ONLY,                 
        datasize, 
        NULL, 
        &status);
    
    //-----------------------------------------------------
    // STEP 6: Write host data to device buffers
    //----------------------------------------------------- 
    
    // Use clEnqueueWriteBuffer() to write input array A to
    // the device buffer bufferA
    status = clEnqueueWriteBuffer(
        cmdQueue, 
        bufferA, 
        CL_FALSE, 
        0, 
        datasize,                         
        A, 
        0, 
        NULL, 
        NULL);
    
    // Use clEnqueueWriteBuffer() to write input array B to 
    // the device buffer bufferB
    status = clEnqueueWriteBuffer(
        cmdQueue, 
        bufferB, 
        CL_FALSE, 
        0, 
        datasize,                                  
        B, 
        0, 
        NULL, 
        NULL);

    //-----------------------------------------------------
    // STEP 7: Create and compile the program
    //----------------------------------------------------- 
     
    // Create a program using clCreateProgramWithSource()
    cl_program program = clCreateProgramWithSource(
        context, 
        1, 
        (const char**)&programSource,                                 
        NULL, 
        &status);

    // Build (compile) the program for the devices with
    // clBuildProgram()
    status = clBuildProgram(
        program, 
        numDevices, 
        devices, 
        NULL, 
        NULL, 
        NULL);
   
    //-----------------------------------------------------
    // STEP 8: Create the kernel
    //----------------------------------------------------- 

    cl_kernel kernel = NULL;

    // Use clCreateKernel() to create a kernel from the 
    // vector addition function (named "vecadd")
    kernel = clCreateKernel(program, "vecadd", &status);

    //-----------------------------------------------------
    // STEP 9: Set the kernel arguments
    //----------------------------------------------------- 
    
    // Associate the input and output buffers with the 
    // kernel 
    // using clSetKernelArg()
    status  = clSetKernelArg(
        kernel, 
        0, 
        sizeof(cl_mem), 
        &bufferA);
    status |= clSetKernelArg(
        kernel, 
        1, 
        sizeof(cl_mem), 
        &bufferB);
    status |= clSetKernelArg(
        kernel, 
        2, 
        sizeof(cl_mem), 
        &bufferC);

    //-----------------------------------------------------
    // STEP 10: Configure the work-item structure
    //----------------------------------------------------- 
    
    // Define an index space (global work size) of work 
    // items for 
    // execution. A workgroup size (local work size) is not 
    // required, 
    // but can be used.
    size_t globalWorkSize[1];    
    // There are 'elements' work-items 
    globalWorkSize[0] = elements;


	double time_opencl = .0;
	double time_host = .0;
	double t1, t2;
    //-----------------------------------------------------
    // STEP 11: Enqueue the kernel for execution
    //----------------------------------------------------- 
    
    // Execute the kernel by using 
    // clEnqueueNDRangeKernel().
    // 'globalWorkSize' is the 1D dimension of the 
    // work-items
	for(int i=0; i<iters; i++){
		t1 = dtime();	
    	status = clEnqueueNDRangeKernel(
        	cmdQueue, 
        	kernel, 
        	1, 
        	NULL, 
        	globalWorkSize, 
        	NULL, 
        	0, 
        	NULL, 
        	NULL);
		clFinish(cmdQueue);
		t2 = dtime();	
		time_opencl += t2 - t1;
	}
	time_opencl = time_opencl/iters;
	
    //-----------------------------------------------------
    // STEP 12: Read the output buffer back to the host
    //----------------------------------------------------- 
    
    // Use clEnqueueReadBuffer() to read the OpenCL output  
    // buffer (bufferC) 
    // to the host output array (C)
    clEnqueueReadBuffer(
        cmdQueue, 
        bufferC, 
        CL_TRUE, 
        0, 
        datasize, 
        C, 
        0, 
        NULL, 
        NULL);

    // Verify the output
    bool result = true;
    for(int i = 0; i < elements; i++) {
        if(C[i] != i+i) {
            result = false;
            break;
        }
    }
    if(result) {
        printf("Output is correct\n");
    } else {
        printf("Output is incorrect\n");
    }

	
    //-----------------------------------------------------
    // STEP :test host version.
    //----------------------------------------------------- 
	for(int i=0; i<iters; i++){
		time_host += hostVectorAdd(C, A, B, elements);	
	}
	time_host = time_host/iters;
	
	printf("time_opencl = %lf us, time_host=%lf us. speedup=%lf. \n", time_opencl, time_host, time_host/time_opencl);
    //-----------------------------------------------------
    // STEP 13: Release OpenCL resources
    //----------------------------------------------------- 
    
    // Free OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseContext(context);

    // Free host resources
    free(A);
    free(B);
    free(C);
    free(platforms);
    free(devices);
}
