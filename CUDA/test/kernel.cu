#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <cuda.h>
#include <stdio.h>

//static float N = 1000;

__global__ void addKernel(float*a, float*b, float*c, float N)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i<N) {
		c[i] =  a[i] + b[i];
	}



}

int main() {
	double *d_a;
	cudaMalloc((void**)&d_a, 1024 * sizeof(double));
	cudaError err = cudaGetLastError();
	if (err == cudaSuccess) {
		printf("double its ok!\n");
	}
	else {
		printf("double its fatal error\n");
	}

	float *d_b;
	int n = 100000;
	cudaMalloc((void**)&d_a, n * sizeof(float));
	err = cudaGetLastError();
	if (err == cudaSuccess) {
		printf("float its ok!\n");
	}
	else {
		printf("float its fatal error\n");
	}

	/*int N = 256 * 256;
	float *h_a, *h_b, *h_res;
	float *d_a, *d_b, *d_res;
	h_a = (float*)malloc(N * sizeof(float));
	h_b = (float*)malloc(N * sizeof(float));
	h_res = (float*)malloc(N * sizeof(float));

	cudaMalloc((void**)&d_a, N * sizeof(float));
	cudaMalloc((void**)&d_b, N * sizeof(float));
	cudaMalloc((void**)&d_res, N * sizeof(float));

	
	for (int i = 0; i < N; i++) {
		h_a[0]= 100;
		h_b[0] = 33;
		h_res[i] = 0;
	}





	cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

	int threads = 256;
	int blocks = 256;

	addKernel << <threads, blocks >> > (d_a, d_b, d_res, N);

	cudaMemcpy(h_res, d_res, N * sizeof(float), cudaMemcpyDeviceToHost);



	for (int i = 0; i < 10; i++) {
		printf("%d  \n", h_res[i]);
	}
	*/
	system("pause");
	return 0;
}