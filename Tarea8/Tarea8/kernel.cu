#include <iostream>
#include <chrono>
#include <cuda.h>
#include <cuda_runtime.h>

// Definición de parámetros para el polinomio
#define DEGREE 1024  // Grado del polinomio (puede cambiarse)
#define THREADS_PER_BLOCK 256  // Hilos por bloque

// Declaración de memoria constante para los coeficientes del polinomio
__constant__ float d_coeffs[DEGREE + 1];

// Kernel para calcular el valor del polinomio en paralelo
__global__ void evaluatePolynomial(float x, float* d_result, int degree) {
    __shared__ float partial_sum[THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float term = 0.0;

    // Cada hilo calcula su término correspondiente
    if (tid <= degree) {
        term = d_coeffs[tid] * powf(x, tid);
    }
    partial_sum[threadIdx.x] = term;
    __syncthreads();

    // Reducción secuencial de los términos calculados
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sum[threadIdx.x] += partial_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Almacenar el resultado final en memoria global desde el hilo 0 del bloque 0
    if (threadIdx.x == 0) {
        atomicAdd(d_result, partial_sum[0]);
    }
}

// Función de inicialización de los coeficientes del polinomio
void initializeCoefficients(float* h_coeffs, int degree) {
    for (int i = 0; i <= degree; ++i) {
        h_coeffs[i] = static_cast<float>(i + 1);  // Coeficientes arbitrarios
    }
}

int main() {
    float x = 1.5f;  // Valor de x para evaluar el polinomio
    int degree = DEGREE;

    // Inicializar los coeficientes del polinomio en el host
    float h_coeffs[DEGREE + 1];
    initializeCoefficients(h_coeffs, degree);

    // Copiar los coeficientes a la memoria constante en el dispositivo
    cudaMemcpyToSymbol(d_coeffs, h_coeffs, (degree + 1) * sizeof(float));

    // Reserva de memoria para el resultado en el dispositivo
    float* d_result;
    cudaMalloc(&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));

    // Definir el número de bloques e hilos
    int blocks = (degree + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Medición del tiempo usando chrono
    auto start_chrono = std::chrono::high_resolution_clock::now();
    evaluatePolynomial << <blocks, THREADS_PER_BLOCK >> > (x, d_result, degree);
    cudaDeviceSynchronize();
    auto end_chrono = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_chrono = end_chrono - start_chrono;

    // Medición del tiempo usando eventos de CUDA
    cudaEvent_t start_cuda, stop_cuda;
    cudaEventCreate(&start_cuda);
    cudaEventCreate(&stop_cuda);
    cudaEventRecord(start_cuda);

    cudaMemset(d_result, 0, sizeof(float));  // Resetear el resultado
    evaluatePolynomial << <blocks, THREADS_PER_BLOCK >> > (x, d_result, degree);
    cudaEventRecord(stop_cuda);
    cudaEventSynchronize(stop_cuda);

    float duration_cuda;
    cudaEventElapsedTime(&duration_cuda, start_cuda, stop_cuda);

    // Copiar el resultado al host y mostrarlo
    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Polynomial result: " << h_result << std::endl;

    // Mostrar los tiempos de ejecución
    std::cout << "Time measured with chrono: " << duration_chrono.count() << " ms" << std::endl;
    std::cout << "Time measured with CUDA events: " << duration_cuda << " ms" << std::endl;

    // Liberar memoria
    cudaFree(d_result);
    cudaEventDestroy(start_cuda);
    cudaEventDestroy(stop_cuda);

    return 0;
}
