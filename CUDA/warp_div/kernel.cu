
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <stdio.h>
#include <math.h>

static int N = 1000000;

#define CHECK(a) {err = cudaGetLastError();\
if( err != cudaSuccess ) \
printf(a);}

__global__ void SomeKernel(float *a, float *b, float *c, int nN)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < nN)
	{
		if (threadIdx.x%2 == 0) {
			c[id] = sin(a[id]) + sin(b[id]);
		}
		else {
			c[id] = sin(a[id]) - sin(b[id]);
		}
		
	}
}

void HostSum(float *a, float *b, float *c, int nN) {
	for (int i = 0; i < N; i++) {
		c[i] = sin(a[i]) + sin(b[i]);
	}
}

int main()
{
	cudaError_t err;
	clock_t start, stop;
	float *h_a, *h_b, *h_res;
	float *d_a, *d_b, *d_res;

	//События
	cudaEvent_t event_start, event_stop, all_start, all_stop;
	float kernelTime = 0.0, allTime = 0.0;
	//создаём события
	cudaEventCreate(&event_start);
	cudaEventCreate(&event_stop);
	cudaEventCreate(&all_start);
	cudaEventCreate(&all_stop);



	// Выделение оперативной памяти (для CPU)
	cudaMallocHost((void**)&h_a, N * sizeof(N));
	cudaMallocHost((void**)&h_b, N * sizeof(N));
	cudaMallocHost((void**)&h_res, N * sizeof(N));

	// Инициализация исходных данных
	for (int i = 0; i < N; ++i)
	{
		h_a[i] = i / (i + 1.0);
		h_b[i] = i / (i + 3.0);
		h_res[i] = 0;
	}
	start = clock();
	HostSum(h_a, h_b, h_res, N);
	stop = clock();
	printf("Time Host sum = %d\n", (stop - start));

	
	// Выделение памяти GPU
	cudaMalloc((void**)&d_a, N * sizeof(float));
	CHECK("cuda malloc 1\n");
	cudaMalloc((void**)&d_b, N * sizeof(float));
	CHECK("cuda malloc 2\n");
	cudaMalloc((void**)&d_res, N * sizeof(float));
	CHECK("cuda malloc 3\n");
	
	cudaEventRecord(all_start, 0);
	cudaMemcpyAsync(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice, 0);
	CHECK("cuda memcpy HostToDevice 1\n");
	cudaMemcpyAsync(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice, 0);
	CHECK("cuda memcpy HostToDevice 2\n");

	int threads = 32;
	int blocks = N / 32;


	
	SomeKernel << <blocks, threads >> > (d_a, d_b, d_res, N);
	CHECK("kernel \n");
	
	cudaMemcpyAsync(h_res, d_res, N * sizeof(float), cudaMemcpyDeviceToHost, 0);
	CHECK("cuda memcpy DeviceToHost \n");


	cudaEventRecord(all_stop, 0);
	cudaEventSynchronize(all_stop);
	cudaEventElapsedTime(&allTime, all_start, all_stop);
	printf("All time Device sum = %f\n", allTime);


	cudaEventDestroy(all_start);
	cudaEventDestroy(all_stop);
	cudaEventDestroy(event_start);
	cudaEventDestroy(event_stop);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);
	return 0;
}