#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <chrono>  

using namespace std;

const int SIGNAL_SIZE = 1024;
const int KERNEL_SIZE = 5;

// Convolución en CPU: toma la señal y el kernel y realiza la convolución en el CPU
void convolutionCPU(const vector<float>& signal, const vector<float>& kernel, vector<float>& output) {
    int halfKernel = KERNEL_SIZE / 2; // Define el centro del kernel
    for (int i = 0; i < SIGNAL_SIZE; i++) {
        float result = 0;
        // Convolucionamos cada elemento de la señal con cada elemento del kernel
        for (int j = 0; j < KERNEL_SIZE; j++) {
            int index = i + j - halfKernel; // Calcular el índice de la señal a partir del centro
            if (index >= 0 && index < SIGNAL_SIZE) { // Verificar que el índice esté dentro de los límites
                result += signal[index] * kernel[j];
            }
        }
        output[i] = result; // Guardar el resultado
    }
}

// Convolución en GPU 
__global__ void convolutionGPU(float* d_signal, float* d_kernel, float* d_output, int signalSize, int kernelSize) {
    extern __shared__ float sharedSignal[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int halfKernel = kernelSize / 2;

    // Copiar datos a memoria compartida incluyendo márgenes adicionales
    if (gid < signalSize) {
        sharedSignal[tid + halfKernel] = d_signal[gid];
    }
    else {
        sharedSignal[tid + halfKernel] = 0.0f; // Padding con ceros si estamos fuera de rango
    }
    if (tid < halfKernel) {  // Cargar márgenes a la izquierda
        sharedSignal[tid] = (gid >= halfKernel) ? d_signal[gid - halfKernel] : 0.0f;
    }
    if (tid >= blockDim.x - halfKernel) {  // Cargar márgenes a la derecha
        sharedSignal[tid + kernelSize - 1] = (gid + halfKernel < signalSize) ? d_signal[gid + halfKernel] : 0.0f;
    }
    __syncthreads();

    // Cálculo de la convolución
    float result = 0;
    if (gid < signalSize) {
        for (int j = 0; j < kernelSize; j++) {
            result += sharedSignal[tid + j] * d_kernel[j];
        }
        d_output[gid] = result;
    }
}


int main() {
    vector<float> h_signal(SIGNAL_SIZE, 1.0f); // Inicializar señal con valor arbitrario
    vector<float> h_kernel(KERNEL_SIZE, 0.2f); // Inicializar kernel con valor arbitrario
    vector<float> h_output(SIGNAL_SIZE, 0); // Vector de salida para la convolución en CPU

    float* d_signal, * d_kernel, * d_output;

    // Reservar memoria en GPU
    cudaMalloc((void**)&d_signal, SIGNAL_SIZE * sizeof(float));
    cudaMalloc((void**)&d_kernel, KERNEL_SIZE * sizeof(float));
    cudaMalloc((void**)&d_output, SIGNAL_SIZE * sizeof(float));

    // Copiar datos de la señal y kernel a la GPU
    cudaMemcpy(d_signal, h_signal.data(), SIGNAL_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256; // Tamaño del bloque
    int gridSize = (SIGNAL_SIZE + blockSize - 1) / blockSize; // Tamaño de la malla

    // Medir el tiempo de ejecución en CPU
    auto start_cpu = chrono::high_resolution_clock::now();
    convolutionCPU(h_signal, h_kernel, h_output);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration_cpu = end_cpu - start_cpu;
    cout << "Tiempo de ejecucion en CPU: " << duration_cpu.count() << " ms" << endl;

    // Medir el tiempo de ejecución en GPU
    auto start_gpu = chrono::high_resolution_clock::now();
    convolutionGPU << <gridSize, blockSize, blockSize * sizeof(float) >> > (d_signal, d_kernel, d_output, SIGNAL_SIZE, KERNEL_SIZE);
    cudaDeviceSynchronize();
    auto end_gpu = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration_gpu = end_gpu - start_gpu;
    cout << "Tiempo de ejecucion en GPU: " << duration_gpu.count() << " ms\n" << endl;

    // Copiar resultados de GPU a CPU para verificación
    vector<float> gpu_output(SIGNAL_SIZE, 0);
    cudaMemcpy(gpu_output.data(), d_output, SIGNAL_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Comparar resultados entre CPU y GPU
    bool match = true;
    for (int i = 0; i < SIGNAL_SIZE; i++) {
        if (abs(gpu_output[i] - h_output[i]) > 1e-5) {
            cout << "Diferencia en el indice " << i << ": CPU=" << h_output[i] << " GPU=" << gpu_output[i] << endl;
            match = false;
        }
    }
    cout << "\nResultados de GPU y CPU " << (match ? "coinciden." : "no coinciden.") << endl;

    // Mostrar los primeros 10 valores de CPU y GPU para inspección
    cout << "\nPrimeros 10 valores de salida:\n";
    for (int i = 0; i < 10; i++) {
        cout << "Index " << i << ": CPU=" << h_output[i] << " GPU=" << gpu_output[i] << endl;
    }

    // Liberar memoria en GPU
    cudaFree(d_signal);
    cudaFree(d_kernel);
    cudaFree(d_output);

    return 0;
}
