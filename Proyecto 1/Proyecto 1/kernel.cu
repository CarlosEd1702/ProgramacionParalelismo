#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include "bitmap_image.hpp"

using namespace std;

// Definimos la estructura cuComplex para los números complejos en GPU
struct cuComplex {
    float r;
    float i;

    __device__ cuComplex(float a, float b) : r(a), i(b) {}

    __device__ float magnitude2() {
        return r * r + i * i;
    }

    __device__ cuComplex operator*(const cuComplex& a) {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    __device__ cuComplex operator+(const cuComplex& a) {
        return cuComplex(r + a.r, i + a.i);
    }
};

// Función para calcular el conjunto de Julia
__device__ int julia(int x, int y, int n) {
    const float scale = 1.5;
    float jx = scale * (float)(n / 2 - x) / (n / 2.f);
    float jy = scale * (float)(n / 2 - y) / (n / 2.f);
    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);
    for (int i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }
    return 1;
}

// Kernel para calcular el fractal en GPU
__global__ void computeJuliaSet(unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        int color = julia(x, y, width);
        image[idx] = color * 255;      // R
        image[idx + 1] = 0;            // G
        image[idx + 2] = color * 255;  // B
    }
}

// Función principal
int main() {
    int width = 800; // Ancho de la imagen
    int height = 800; // Alto de la imagen

    // Crear una imagen vacía en CPU
    bitmap_image image(width, height);

    // Espacio de memoria en GPU
    unsigned char* dev_image;
    size_t image_size = width * height * 3 * sizeof(unsigned char);

    // Asignar memoria en el dispositivo (GPU)
    cudaMalloc((void**)&dev_image, image_size);

    // Definir el tamaño de bloques y grid
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Llamar al kernel en GPU
    computeJuliaSet << <blocksPerGrid, threadsPerBlock >> > (dev_image, width, height);

    // Crear un buffer para recibir la imagen del GPU
    unsigned char* host_image = (unsigned char*)malloc(image_size);

    // Copiar resultados del device (GPU) al host (CPU)
    cudaMemcpy(host_image, dev_image, image_size, cudaMemcpyDeviceToHost);

    // Guardar la imagen en el CPU usando bitmap_image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            image.set_pixel(x, y, host_image[idx], host_image[idx + 1], host_image[idx + 2]);
        }
    }
    image.save_image("julia_fractal.bmp");

    // Liberar memoria
    cudaFree(dev_image);
    free(host_image);

    return 0;
}
