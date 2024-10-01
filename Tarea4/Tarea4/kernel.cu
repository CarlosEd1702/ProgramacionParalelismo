#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// Multiplicación de matrices en CPU
void matrixMulCPU(int* A, int* B, int* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0;
            for (int k = 0; k < K; ++k) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

// Multiplicación de matrices en GPU, versión 1: un thread por bloque
__global__ void matrixMulKernel1(int* A, int* B, int* C, int M, int K, int N) {
    int row = blockIdx.y;
    int col = blockIdx.x;
    if (row < M && col < N) {
        int value = 0;
        for (int i = 0; i < K; i++) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

// Versión 2: Un bloque de tamaño uno y threads bidimensionales
__global__ void matrixMulKernel2(int* A, int* B, int* C, int M, int K, int N) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < M && col < N) {
        int value = 0;
        for (int i = 0; i < K; i++) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

// Versión 3: Más de un bloque por grid y más de un thread por bloque
__global__ void matrixMulKernel3(int* A, int* B, int* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int value = 0;
        for (int i = 0; i < K; i++) {
            value += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = value;
    }
}

int main() {
    int M, K, N;

    // Entrada del usuario para las dimensiones de las matrices
    cout << "Ingrese el numero de filas de la matriz A (M): ";
    cin >> M;
    cout << "Ingrese el numero de columnas de la matriz A y filas de B (K): ";
    cin >> K;
    cout << "Ingrese el numero de columnas de la matriz B (N): ";
    cin >> N;

    // Asignar memoria para las matrices en CPU
    int* h_A = (int*)malloc(M * K * sizeof(int));
    int* h_B = (int*)malloc(K * N * sizeof(int));
    int* h_C_cpu = (int*)malloc(M * N * sizeof(int));
    int* h_C_gpu = (int*)malloc(M * N * sizeof(int));

    // Inicializar matrices con valores aleatorios
    srand(time(0));
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = rand() % 10;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = rand() % 10;
    }

    // Multiplicación en CPU
    auto startCPU = chrono::high_resolution_clock::now();
    matrixMulCPU(h_A, h_B, h_C_cpu, M, K, N);
    auto endCPU = chrono::high_resolution_clock::now();
    chrono::duration<float> durationCPU = endCPU - startCPU;
    cout << "Tiempo de ejecucion en CPU: " << durationCPU.count() << " segundos" << endl;

    // Reservar memoria en GPU
    int* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, M * K * sizeof(int));
    cudaMalloc((void**)&d_B, K * N * sizeof(int));
    cudaMalloc((void**)&d_C, M * N * sizeof(int));

    // Copiar datos de CPU a GPU
    cudaMemcpy(d_A, h_A, M * K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(int), cudaMemcpyHostToDevice);

    // Versión 1: Un grid de bloques bidimensional, un thread por bloque
    dim3 grid1(N, M);
    auto startGPU1 = chrono::high_resolution_clock::now();
    matrixMulKernel1 << <grid1, 1 >> > (d_A, d_B, d_C, M, K, N);
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    auto endGPU1 = chrono::high_resolution_clock::now();
    chrono::duration<float> durationGPU1 = endGPU1 - startGPU1;
    cout << "Tiempo de ejecucion en GPU (Version 1): " << durationGPU1.count() << " segundos" << endl;

    // Versión 2: Un bloque de tamaño 1, threads bidimensionales
    dim3 block2(N, M);
    auto startGPU2 = chrono::high_resolution_clock::now();
    matrixMulKernel2 << <1, block2 >> > (d_A, d_B, d_C, M, K, N);
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    auto endGPU2 = chrono::high_resolution_clock::now();
    chrono::duration<float> durationGPU2 = endGPU2 - startGPU2;
    cout << "Tiempo de ejecucion en GPU (Version 2): " << durationGPU2.count() << " segundos" << endl;

    // Versión 3: Más de un bloque por grid y más de un thread por bloque
    dim3 block3(16, 16);  // Tamaño de bloque de 16x16 threads
    dim3 grid3((N + 15) / 16, (M + 15) / 16);  // Tamaño de grid adaptado al tamaño de los bloques
    auto startGPU3 = chrono::high_resolution_clock::now();
    matrixMulKernel3 << <grid3, block3 >> > (d_A, d_B, d_C, M, K, N);
    cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(int), cudaMemcpyDeviceToHost);
    auto endGPU3 = chrono::high_resolution_clock::now();
    chrono::duration<float> durationGPU3 = endGPU3 - startGPU3;
    cout << "Tiempo de ejecucion en GPU (Version 3): " << durationGPU3.count() << " segundos" << endl;

    // Liberar memoria en GPU y CPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);

    return 0;
}
