
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <stdio.h>

static int N = 1000000;

#define CHECK(a) {err = cudaGetLastError();\
if( err != cudaSuccess ) \
printf(a);}

__global__ void SomeKernel(float *a, float *b, float *c, int lN)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < lN)
	{
		c[id] = a[id] + b[id];
	}
}

void HostSum(float *a, float *b, float *&c, int lN) {
	for (int i = 0; i < lN; i++) {
		c[i] = a[i] + b[i];
	}
}

int main()
{
	cudaError_t err;
	clock_t start, stop;
	float *h_a, *h_b, *h_res;
	float *d_a, *d_b, *d_res;

	//События
	cudaEvent_t  all_start, all_stop;
	float  allTime = 0.0;
	//создаём события
	cudaEventCreate(&all_start);
	cudaEventCreate(&all_stop);

	cudaStream_t t[2];


	// Выделение оперативной памяти (для CPU)
	cudaMallocHost((void**)&h_a, N * sizeof(float));
	cudaMallocHost((void**)&h_b, N * sizeof(float));
	cudaMallocHost((void**)&h_res, N * sizeof(float));

	

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
	
	for (int i = 0; i < 2; i++) {
		cudaStreamCreate(&t[i]);
	}

	cudaEventRecord(all_start, 0);
	for (int i = 0; i < 2; i++) {
		cudaMemcpyAsync(d_a + i*(N/2), h_a + i*(N/2), (N/2) * sizeof(float), cudaMemcpyHostToDevice, t[i]);
	}
	CHECK("cuda memcpy HostToDevice 2\n");

	int threads = 32;
	int blocks = (N/2) / 32;

	for (int i = 0; i < 2; i++) {
		SomeKernel << <blocks, threads, 0, t[i] >> > (d_a + i*(N/2), d_b+i*(N/2), d_res+i*(N/2), N/2);
	}
	CHECK("kernel \n");

	for (int i = 0; i < 2; i++) {
		cudaMemcpyAsync(h_res + i*(N/2), d_res + i*(N/2), (N/2) * sizeof(float), cudaMemcpyDeviceToHost, t[i]);
	}
	CHECK("cuda memcpy DeviceToHost \n");

	cudaEventRecord(all_stop, 0);
	cudaEventSynchronize(all_stop);
	cudaEventElapsedTime(&allTime, all_start, all_stop);
	
	printf("All time Device sum = %f\n", allTime);


	cudaEventDestroy(all_start);
	cudaEventDestroy(all_stop);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_res);
	return 0;
}