#include "bitmap_image.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#define CUDA_CHECK(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return EXIT_FAILURE; \
    }

const int WIDTH = 1024;
const int HEIGHT = 1024;
const int ITERATIONS = 100;

__global__ void gameOfLifeKernel(bool* board, bool* nextBoard, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int count = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;

            int nx = (x + dx + width) % width;
            int ny = (y + dy + height) % height;

            count += board[ny * width + nx];
        }
    }

    nextBoard[y * width + x] = (count == 3) || (board[y * width + x] && count == 2);
}

void saveBoardAsImage(bool* board, int width, int height, const std::string& filename) {
    bitmap_image image(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char color = board[y * width + x] ? 255 : 0;
            image.set_pixel(x, y, color, color, color);
        }
    }
    image.save_image(filename);
}

int main() {
    bool* board;
    bool* nextBoard;

    CUDA_CHECK(cudaMallocManaged(&board, WIDTH * HEIGHT * sizeof(bool)));
    CUDA_CHECK(cudaMallocManaged(&nextBoard, WIDTH * HEIGHT * sizeof(bool)));

    // Inicializa el tablero de manera random
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        board[i] = rand() % 2;
    }

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, (HEIGHT + blockSize.y - 1) / blockSize.y);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    for (int i = 0; i < ITERATIONS; ++i) {
        // Despliega el kernel en el stream
        gameOfLifeKernel << <gridSize, blockSize, 0, stream >> > (board, nextBoard, WIDTH, HEIGHT);

        // CPU esta guardando la imagen al mismo tiempo que la GPU esta calculando la siguiente generacion
        if (i % 10 == 0) {
            CUDA_CHECK(cudaStreamSynchronize(stream)); // Asegura que se termine todo el stream 
            saveBoardAsImage(board, WIDTH, HEIGHT, "state_" + std::to_string(i) + ".bmp");
        }

        std::swap(board, nextBoard);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream)); // Bandera para Sincronizar el stream y destruirlo posteriormente
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(board));
    CUDA_CHECK(cudaFree(nextBoard));

    return EXIT_SUCCESS;
}
