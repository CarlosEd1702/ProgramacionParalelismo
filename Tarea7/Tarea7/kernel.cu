#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

// Inicialización de la operación de reducción
#define VECTOR_SIZE 1024

// Versión 1: reducción usando únicamente la operación atómica
__global__ void atomicReduction(float* d_input, float* d_result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        atomicAdd(d_result, d_input[idx]);
    }
}

// Versión 2: reducción usando Sequential Addressing
__global__ void sequentialReduction(float* d_input, float* d_result, int size) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Carga en memoria compartida
    sharedData[tid] = (idx < size) ? d_input[idx] : 0.0f;
    __syncthreads();

    // Reducción en pares, dividiendo el número de hilos a la mitad en cada iteración
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Almacena el resultado en d_result solo si es el hilo principal
    if (tid == 0) atomicAdd(d_result, sharedData[0]);
}

// Versión 3: combinación de Sequential Addressing y operación atómica
__global__ void atomicSequentialReduction(float* d_input, float* d_result, int size) {
    extern __shared__ float sharedData[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Carga en memoria compartida
    sharedData[tid] = (idx < size) ? d_input[idx] : 0.0f;
    __syncthreads();

    // Reducción en pares, dividiendo el número de hilos a la mitad en cada iteración
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Operación atómica en el hilo principal
    if (tid == 0) atomicAdd(d_result, sharedData[0]);
}

// Función de inicialización del vector
void initializeVector(float* h_input, int size) {
    for (int i = 0; i < size; ++i) {
        h_input[i] = 1.0f;  // Asignación de valores para simplificar verificación
    }
}

// Función para ejecutar la reducción y medir el tiempo
void executeReduction(void (*kernel)(float*, float*, int), float* d_input, float* d_result, int size, int threadsPerBlock, const char* versionName) {
    cudaMemset(d_result, 0, sizeof(float));
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    auto start = chrono::high_resolution_clock::now();
    kernel << <blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float) >> > (d_input, d_result, size);
    cudaDeviceSynchronize();
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<float, std::milli> duration = end - start;
    cout << versionName << " Time: " << duration.count() << " ms" << endl;
}

int main() {
    float* h_input, * d_input, * d_result;
    float h_result_cpu = 0.0f, h_result_gpu;

    // Reserva de memoria en el host
    h_input = (float*)malloc(VECTOR_SIZE * sizeof(float));
    initializeVector(h_input, VECTOR_SIZE);

    // Reserva de memoria en el dispositivo
    cudaMalloc((void**)&d_input, VECTOR_SIZE * sizeof(float));
    cudaMalloc((void**)&d_result, sizeof(float));

    // Copia del vector de entrada al dispositivo
    cudaMemcpy(d_input, h_input, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Ejecución de la versión 1: solo operación atómica
    executeReduction(atomicReduction, d_input, d_result, VECTOR_SIZE, 256, "Atomic Only");
    cudaMemcpy(&h_result_gpu, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "Result Atomic Only: " << h_result_gpu << endl;
    h_result_cpu = h_result_gpu;

    // Ejecución de la versión 2: sequential addressing
    executeReduction(sequentialReduction, d_input, d_result, VECTOR_SIZE, 256, "Sequential Addressing");
    cudaMemcpy(&h_result_gpu, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "Result Sequential Addressing: " << h_result_gpu << endl;

    // Verificación de la precisión
    if (h_result_cpu != h_result_gpu) {
        cerr << "Mismatch in Sequential Addressing result!" << endl;
    }

    // Ejecución de la versión 3: combinación de sequential addressing y operación atómica
    executeReduction(atomicSequentialReduction, d_input, d_result, VECTOR_SIZE, 256, "Atomic + Sequential Addressing");
    cudaMemcpy(&h_result_gpu, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cout << "Result Atomic + Sequential Addressing: " << h_result_gpu << endl;

    // Verificación de la precisión
    if (h_result_cpu != h_result_gpu) {
        cerr << "Mismatch in Atomic + Sequential Addressing result!" << endl;
    }

    // Liberación de memoria
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_result);

    return 0;
}
