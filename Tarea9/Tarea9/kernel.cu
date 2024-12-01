#include "bitmap_image.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

#define BLOCK_SIZE 16

// Cargar y guardar BMP
vector<unsigned char> loadBMP(const char* filename, int& width, int& height) {
    bitmap_image img(filename);
    if (!img) {
        cerr << "Error al abrir la imagen BMP." << endl;
        exit(1);
    }

    width = img.width();
    height = img.height();
    vector<unsigned char> img_data(width * height * 3);  // RGB

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            rgb_t color;
            img.get_pixel(x, y, color);
            int idx = (y * width + x) * 3;
            img_data[idx] = color.red;
            img_data[idx + 1] = color.green;
            img_data[idx + 2] = color.blue;
        }
    }
    return img_data;
}

void saveBMP(const char* filename, const vector<unsigned char>& img_data, int width, int height) {
    bitmap_image img(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            rgb_t color;
            color.red = img_data[idx];
            color.green = img_data[idx + 1];
            color.blue = img_data[idx + 2];
            img.set_pixel(x, y, color);
        }
    }
    img.save_image(filename);
}

// Kernel de CUDA para convolución
__global__ void convolutionKernel(const unsigned char* d_input, unsigned char* d_output, int width, int height, const float* d_kernel, int kernelSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int halfKernel = kernelSize / 2;

    if (x >= width || y >= height) return;

    for (int channel = 0; channel < 3; ++channel) {
        float sum = 0.0f;
        for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
            for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
                int ix = min(max(x + kx, 0), width - 1);
                int iy = min(max(y + ky, 0), height - 1);
                int imgIdx = (iy * width + ix) * 3 + channel;
                int kernelIdx = (ky + halfKernel) * kernelSize + (kx + halfKernel);
                sum += d_input[imgIdx] * d_kernel[kernelIdx];
            }
        }
        int outputIdx = (y * width + x) * 3 + channel;
        d_output[outputIdx] = min(max(int(sum), 0), 255);
    }
}

// Función para aplicar la convolución en la GPU
vector<unsigned char> applyConvolutionGPU(const vector<unsigned char>& input, int width, int height, const vector<float>& kernel, int kernelSize) {
    unsigned char* d_input, * d_output;
    float* d_kernel;
    int imgSize = width * height * 3;
    int kernelBytes = kernelSize * kernelSize * sizeof(float);

    cudaMalloc(&d_input, imgSize * sizeof(unsigned char));
    cudaMalloc(&d_output, imgSize * sizeof(unsigned char));
    cudaMalloc(&d_kernel, kernelBytes);

    cudaMemcpy(d_input, input.data(), imgSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel.data(), kernelBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    convolutionKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, d_kernel, kernelSize);

    vector<unsigned char> output(imgSize);
    cudaMemcpy(output.data(), d_output, imgSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);

    return output;
}

// Función para aplicar la convolución en la CPU
vector<unsigned char> applyConvolutionCPU(const vector<unsigned char>& input, int width, int height, const vector<float>& kernel, int kernelSize) {
    vector<unsigned char> output(width * height * 3);
    int halfKernel = kernelSize / 2;
    const float* kernelPtr = kernel.data(); // Memoria constante simulada

#pragma omp parallel for collapse(2) // Uso de paralelismo CPU (opcional)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                float sum = 0.0f;

                // Simulación de memoria compartida para una región de trabajo
                for (int ky = -halfKernel; ky <= halfKernel; ++ky) {
                    for (int kx = -halfKernel; kx <= halfKernel; ++kx) {
                        int ix = min(max(x + kx, 0), width - 1);
                        int iy = min(max(y + ky, 0), height - 1);
                        int imgIdx = (iy * width + ix) * 3 + channel;
                        int kernelIdx = (ky + halfKernel) * kernelSize + (kx + halfKernel);
                        sum += input[imgIdx] * kernelPtr[kernelIdx]; // Acceso optimizado
                    }
                }

                int outputIdx = (y * width + x) * 3 + channel;
                output[outputIdx] = min(max(int(sum), 0), 255); // Clamp de valores
            }
        }
    }

    return output;
}

int main() {
    const char* inputFilename = "canvas.bmp";
    const char* outputFilenameGPU = "output_gpu_Tarea9.bmp";
    const char* outputFilenameCPU = "output_cpu_Tarea9.bmp";
    int width, height;

    // Cargar imagen
    vector<unsigned char> inputImage = loadBMP(inputFilename, width, height);

    // Definir un kernel 
    vector<float> kernel = {
      -2.0f, -1.0f, 0.0f,
      -1.0f, 0.0f, 1.0f,
      0.0f, 1.0f, 2.0f,
    };
    int kernelSize = 3;


    // Aplicar convolución en GPU y medir tiempo
    auto startGPU = chrono::high_resolution_clock::now();
    vector<unsigned char> outputImageGPU = applyConvolutionGPU(inputImage, width, height, kernel, kernelSize);
    auto endGPU = chrono::high_resolution_clock::now();
    chrono::duration<double> gpuTime = endGPU - startGPU;

    // Guardar la imagen resultante de la GPU
    saveBMP(outputFilenameGPU, outputImageGPU, width, height);

    // Aplicar convolución en CPU y medir tiempo
    auto startCPU = chrono::high_resolution_clock::now();
    vector<unsigned char> outputImageCPU = applyConvolutionCPU(inputImage, width, height, kernel, kernelSize);
    auto endCPU = chrono::high_resolution_clock::now();
    chrono::duration<double> cpuTime = endCPU - startCPU;

    // Guardar la imagen resultante de la CPU
    saveBMP(outputFilenameCPU, outputImageCPU, width, height);

    // Mostrar tiempos de ejecución
    cout << outputFilenameGPU << " Tiempo de ejecucion en GPU: " << gpuTime.count() << " segundos" << endl;
    cout << outputFilenameCPU << " Tiempo de ejecucion en CPU: " << cpuTime.count() << " segundos" << endl;

    return 0;
}