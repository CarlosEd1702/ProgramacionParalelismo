#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "bitmap_image.hpp"

const int WIDTH = 1024;
const int HEIGHT = 1024;
const int ITERATIONS = 100;

// Macro para verificar errores de CUDA
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in " << __FILE__ << " at "   \
                      << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel para ejecutar el Juego de la Vida
__global__ void gameOfLifeKernel(int* current, int* next, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Calcular el número de vecinos vivos
    int neighbors = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                neighbors += current[ny * width + nx];
            }
        }
    }

    // Aplicar las reglas del juego
    if (current[idx] == 1) {
        next[idx] = (neighbors == 2 || neighbors == 3) ? 1 : 0;
    }
    else {
        next[idx] = (neighbors == 3) ? 1 : 0;
    }
}

// Función para guardar el tablero como una imagen BMP
void saveBoardAsImage(const int* board, int width, int height, const std::string& filename) {
    bitmap_image image(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char color = board[y * width + x] ? 255 : 0;
            image.set_pixel(x, y, color, color, color);
        }
    }
    image.save_image(filename);
}

// Función para inicializar el tablero con un patrón aleatorio
void initializeBoard(int* board, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        board[i] = rand() % 2;
    }
}

int main() {
    // Asignar memoria unificada para los tableros
    int* board;
    int* nextBoard;
    CUDA_CHECK(cudaMallocManaged(&board, WIDTH * HEIGHT * sizeof(int)));
    CUDA_CHECK(cudaMallocManaged(&nextBoard, WIDTH * HEIGHT * sizeof(int)));

    // Inicializar el tablero
    initializeBoard(board, WIDTH, HEIGHT);
    saveBoardAsImage(board, WIDTH, HEIGHT, "initial_state.bmp");

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    // Ejecutar las iteraciones
    for (int i = 0; i < ITERATIONS; ++i) {
        gameOfLifeKernel << <gridSize, blockSize >> > (board, nextBoard, WIDTH, HEIGHT);
        CUDA_CHECK(cudaDeviceSynchronize());

        if (i % 10 == 0) {  // Guardar estados intermedios
            saveBoardAsImage(board, WIDTH, HEIGHT, "state_" + std::to_string(i) + ".bmp");
        }

        std::swap(board, nextBoard);
    }

    // Guardar el estado final
    saveBoardAsImage(board, WIDTH, HEIGHT, "final_state.bmp");

    // Liberar memoria
    CUDA_CHECK(cudaFree(board));
    CUDA_CHECK(cudaFree(nextBoard));

    return 0;
}
