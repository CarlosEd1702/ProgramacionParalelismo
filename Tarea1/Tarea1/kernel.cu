#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void cuda_hello() {
	printf("Hello World from GPU\n");
}

void add_cpu(int* a, int* b, int* c) {

	*c = *a + *b;
}

__global__ void add_gpu(int *a, int *b, int *c) {
	
	*c = *a + *b;
	
	/*int example;
	int* ptr_example = &example;
	int** ptr_ptr_example = &ptr_example;

	int example2 = *ptr_example;*/
}

int main() {
	int a = 2;
	int* b;
	int c;
	b = &c;
	*b = 5;
	c = 5;
	int result;

	float a_float = (float)a;
	char letra_a = 'a';

	int* a_gpu, *b_gpu, *result_gpu;

	int size = sizeof(int);
	int size_a = sizeof(a);



	//cout << "letra a: " << letra_a << " letra_a a int " << (int)letra_a << endl;

	add_cpu(&a, b, &result);
	add_gpu << <1, 1 >> > (&a, b, &result);
	
	cout << "El resultado de la suma en CPU es: " << result << "\n";

	// cout << "Int: " << size << " size_a: " << size_a << endl;
	//cuda_hello <<<1, 10>>> ();

	/*cudaMalloc((void**)&a_gpu, size);
	cudaMalloc((void**)&b_gpu, size);
	cudaMalloc((void**)&result_gpu, size);

	cudaMemcpy(a_gpu, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(b_gpu, &c, size, cudaMemcpyHostToDevice);
	
	cudaMemcpy(&result, result_gpu, size, cudaMemcpyDeviceToHost);

	cout << "Result " << result << endl;
	printf("Result %d", result);

	cudaFree(a_gpu);
	cudaFree(b_gpu);
	cudaFree(result_gpu);*/

	return 0;
}

//int main() {
//	cout << "Hello World!" << endl;
//	return 0;
//}
