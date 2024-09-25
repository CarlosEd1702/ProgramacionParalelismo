﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <chrono>

using namespace std;

// Multiplicación de matrices en GPU 
__global__ void matrixMulKernel(int* d_A, int* d_B, int* d_C, int N, int M, int K) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            d_C[i * K + j] = 0;
            for (int k = 0; k < M; ++k) {
                d_C[i * K + j] += d_A[i * M + k] * d_B[k * K + j];
            }
        }
    }
}

// Multiplicación de matrices en CPU 
void matrixMulCPU(int* A, int* B, int* C, int N, int M, int K) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            C[i * K + j] = 0;
            for (int k = 0; k < M; ++k) {
                C[i * K + j] += A[i * M + k] * B[k * K + j];
            }
        }
    }
}

int main()
{
    int N, M, K;

    // Entrada de dimensiones de las matrices
    cout << "Ingrese el numero de filas de la matriz A (N): ";
    cin >> N;
    cout << "Ingrese el numero de columnas de la matriz A y filas de B (M): ";
    cin >> M;
    cout << "Ingrese el numero de columnas de la matriz B (K): ";
    cin >> K;

    // Reservar memoria para las matrices en el host
    int* h_A = (int*)malloc(N * M * sizeof(int));
    int* h_B = (int*)malloc(M * K * sizeof(int));
    int* h_C = (int*)malloc(N * K * sizeof(int));  // Resultado en CPU
    int* h_C_gpu = (int*)malloc(N * K * sizeof(int));  // Resultado en GPU

    // Inicializar matrices con valores aleatorios
    srand(time(0));
    for (int i = 0; i < N * M; ++i) {
        h_A[i] = rand() % 10;
    }
    for (int i = 0; i < M * K; ++i) {
        h_B[i] = rand() % 10;
    }

    auto start = chrono::system_clock::now();

    // Multiplicación de matrices en la CPU
    matrixMulCPU(h_A, h_B, h_C, N, M, K);

    // Mostrar resultado de la multiplicación en CPU
    cout << "\nResultado en CPU:" << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            cout << h_C[i * K + j] << " ";
        }
        cout << endl;
    }

    auto end = chrono::system_clock::now();
    chrono::duration<float, milli> duration = end - start;
    cout << "\nTiempo de CPU(chrono): " << duration.count() << " ms" << endl;

    // Variables en el dispositivo (GPU)
    int* d_A, * d_B, * d_C;

    // Asignar memoria en el dispositivo
    cudaMalloc((void**)&d_A, N * M * sizeof(int));
    cudaMalloc((void**)&d_B, M * K * sizeof(int));
    cudaMalloc((void**)&d_C, N * K * sizeof(int));

    // Copiar datos de host a device
    cudaMemcpy(d_A, h_A, N * M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, M * K * sizeof(int), cudaMemcpyHostToDevice);

    // Establecer el número de hilos
    int numThreads = N * K;

    auto startGPU = chrono::system_clock::now();

    // Multiplicación de matrices en la GPU
    matrixMulKernel << <1, numThreads >> > (d_A, d_B, d_C, N, M, K);

    // Copiar el resultado de vuelta a host
    cudaMemcpy(h_C_gpu, d_C, N * K * sizeof(int), cudaMemcpyDeviceToHost);

    // Mostrar resultado de la multiplicación en GPU 
    cout << "\nResultado en GPU:" << endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            cout << h_C_gpu[i * K + j] << " ";
        }
        cout << endl;
    }

    auto endGPU = chrono::system_clock::now();
    chrono::duration<float, milli> durationGPU = endGPU - startGPU;

    cout << "\nTiempo de GPU(chrono): " << durationGPU.count() << " ms" << endl;

    // Liberar memoria en GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Liberar memoria en host
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_gpu);

    return 0;
}