#include<stdio.h>
#include<stdlib.h>

#ifdef _APPLE_
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

void checkErr(cl_int err, int num){
	if(CL_SUCCESS != err){
		printf("OpenCL error(%d) at %d\n", err, num  - 1);
	}
}

void platformInfo(){

	cl_platform_id *platform;
	cl_uint num_platform;
	cl_int err;

	err = clGetPlatformIDs(0, NULL,  &num_platform);
	platform = (cl_platform_id *)malloc(sizeof (cl_platform_id) * num_platform);
	err = clGetPlatformIDs(num_platform, platform,  NULL);

	printf("numofPlatform=%d.\n", num_platform);

	for(int i = 0; i < num_platform; i++){
		size_t size;
		err = clGetPlatformInfo(platform[i],  CL_PLATFORM_NAME, 0, NULL, &size);
		
		char *PName = (char *)malloc(size);
		err = clGetPlatformInfo(platform[i],  CL_PLATFORM_NAME, size, PName, NULL);
		printf("\nCL_PLATFORM_NAME: % s\n", PName);
		
		err = clGetPlatformInfo(platform[i],  CL_PLATFORM_VENDOR, 0, NULL, &size);

		char *PVendor = (char *)malloc(size);
		err = clGetPlatformInfo(platform[i],  CL_PLATFORM_VENDOR, size, PVendor,  NULL);
		printf("CL_PLATFORM_VENDOR: % s\n",  PVendor);

		err = clGetPlatformInfo(platform[i],  CL_PLATFORM_VERSION, 0, NULL, &size);
		char *PVersion = (char *)malloc(size);
		err = clGetPlatformInfo(platform[i],  CL_PLATFORM_VERSION, size, PVersion,  NULL);
		printf("CL_PLATFORM_VERSION: % s\n",  PVersion);
		err = clGetPlatformInfo(platform[i],  CL_PLATFORM_PROFILE, 0, NULL, &size);

		char *PProfile = (char *)malloc(size);
		err = clGetPlatformInfo(platform[i],  CL_PLATFORM_PROFILE,	size, PProfile,  NULL);
		printf("CL_PLATFORM_PROFILE: % s\n",  PProfile);

		err = clGetPlatformInfo(platform[i],  CL_PLATFORM_EXTENSIONS, 0, NULL, &size);

		char *PExten = (char *)malloc(size);
		err = clGetPlatformInfo(platform[i],  CL_PLATFORM_EXTENSIONS, size, PExten, NULL);
		printf("CL_PLATFORM_EXTENSIONS: % s\n",  PExten);
	
		free(PName);
		free(PVendor);
		free(PVersion);
		free(PProfile);
		free(PExten);
	}
	free(platform);	
}


int main(int argc, char **argv){

	cl_device_id *device;
	cl_platform_id platform;
	cl_int err;
	cl_uint NumDevice;


	platformInfo();	

	//3-7: 选择第一个平台
	err = clGetPlatformIDs(1, &platform, NULL);
	checkErr(err, __LINE__);

	err = clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU, 0, NULL, &NumDevice);
	checkErr(err, __LINE__);

	device = (cl_device_id *)malloc(sizeof (cl_device_id) * NumDevice);
	
	//选择GPU设备
	err = clGetDeviceIDs(platform,  CL_DEVICE_TYPE_GPU, NumDevice, device, NULL);
	checkErr(err, __LINE__);


	printf("NumDevice=%d\n", NumDevice);

for(int i = 0; i < NumDevice; i++){
	//查询设备名称
	char buffer[100];
	err = clGetDeviceInfo(device[i],  CL_DEVICE_NAME, 100, buffer, NULL);
	checkErr(err, __LINE__);
	printf("Device Name:%s\n", buffer);
	//查询设备计算单元最大数目
	cl_uint UnitNum;
	err = clGetDeviceInfo(device[i],
	CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint),  &UnitNum, NULL);
	checkErr(err, __LINE__);
	printf("Compute Units Number: %d\n",  UnitNum);
	
	//查询设备核心频率
	cl_uint frequency;
	err = clGetDeviceInfo(device[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint),  &frequency, NULL);
	checkErr(err, __LINE__);
	printf("Device Frequency: %d(MHz)\n",  frequency);

	//查询设备全局内存大小
	cl_ulong GlobalSize;
	err = clGetDeviceInfo(device[i],  CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),  &GlobalSize, NULL);
	checkErr(err, __LINE__);
	printf("Device Global Size: %0.0f(MB)\n", (float)GlobalSize / 1024 / 1024);
	
	//查询设备全局内存缓存行
	cl_uint GlobalCacheLine;
	err = clGetDeviceInfo(device[i], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint),  &GlobalCacheLine, NULL);
	checkErr(err, __LINE__);
	printf("Device Global CacheLine: %d (Byte)\n",	GlobalCacheLine);

	//查询设备支持的OpenCL版本
	char DeviceVersion[100];
	err = clGetDeviceInfo(device[i], CL_DEVICE_VERSION, 100, DeviceVersion,  NULL);
	checkErr(err, __LINE__);
	printf("Device Version:%s\n",  DeviceVersion);

	//查询设备拓展名
	char *DeviceExtensions;
	//cl_uint ExtenNum;
	size_t ExtenNum;
	err = clGetDeviceInfo(device[i], CL_DEVICE_EXTENSIONS, 0, NULL,  &ExtenNum);
	checkErr(err, __LINE__);

	DeviceExtensions = (char *)malloc(ExtenNum);
	err = clGetDeviceInfo(device[i],  CL_DEVICE_EXTENSIONS, ExtenNum,  DeviceExtensions, NULL);
	checkErr(err, __LINE__);
	printf("Device Extensions:%s\n",  DeviceExtensions);

	free(DeviceExtensions);
}

free(device);
return 0;
}
