#include <iostream>
#include <cuda_runtime.h>

using namespace std;

void printDeviceProperties(int device) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    cout << "Propiedades del dispositivo (GPU):\n";
    cout << "-----------------------------------\n";

    // Nombre del dispositivo
    cout << "Nombre del dispositivo: " << prop.name << "\n";
    cout << "  -> Nombre de la GPU\n\n";

    // Memoria global total
    cout << "Memoria global total: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    cout << "  -> Memoria total accesible por la GPU\n\n";

    // Memoria compartida por bloque
    cout << "Memoria compartida por bloque: " << prop.sharedMemPerBlock / 1024.0 << " KB\n";
    cout << "  -> Memoria compartida disponible por cada bloque\n\n";

    // Registros por bloque
    cout << "Registros por bloque: " << prop.regsPerBlock << "\n";
    cout << "  -> Cantidad de registros disponibles por bloque\n\n";

    // Tamaño de warp
    cout << "Tamaño de warp: " << prop.warpSize << "\n";
    cout << "  -> Número de threads en un warp (unidad de ejecución)\n\n";

    // Paso máximo de memoria (pitch)
    cout << "Paso máximo de memoria (memPitch): " << prop.memPitch << " bytes\n";
    cout << "  -> Máximo tamaño de la fila de una matriz en memoria lineal\n\n";

    // Máximo de threads por bloque
    cout << "Máximo de threads por bloque: " << prop.maxThreadsPerBlock << "\n";
    cout << "  -> Máximo número de threads que puede contener un bloque\n\n";

    // Dimensiones máximas de un bloque
    cout << "Dimensiones máximas de un bloque: (" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")\n";
    cout << "  -> Máximo número de threads en cada dimensión de un bloque\n\n";

    // Tamaño máximo del grid
    cout << "Tamaño máximo del grid: (" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")\n";
    cout << "  -> Tamaño máximo del grid (en bloques) en cada dimensión\n\n";

    // Memoria constante total
    cout << "Memoria constante total: " << prop.totalConstMem / 1024.0 << " KB\n";
    cout << "  -> Tamaño de la memoria constante disponible\n\n";

    // Capacidad computacional (major.minor)
    cout << "Capacidad computacional: " << prop.major << "." << prop.minor << "\n";
    cout << "  -> Versión de la arquitectura CUDA\n\n";

    // Velocidad del reloj
    cout << "Velocidad del reloj: " << prop.clockRate / 1000 << " MHz\n";
    cout << "  -> Frecuencia máxima de reloj de la GPU\n\n";

    // Alineación de la textura
    cout << "Alineación de la textura: " << prop.textureAlignment << " bytes\n";
    cout << "  -> Alineación requerida para la memoria de texturas\n\n";

    // Device overlap
    cout << "Device overlap: " << (prop.deviceOverlap ? "Si" : "No") << "\n";
    cout << "  -> Si la GPU puede ejecutar copias de memoria y cálculos en paralelo\n\n";

    // Cantidad de multiprocesadores
    cout << "Cantidad de multiprocesadores: " << prop.multiProcessorCount << "\n";
    cout << "  -> Número de multiprocesadores en la GPU\n\n";

    // Kernel Execution Timeout
    cout << "Kernel Execution Timeout: " << (prop.kernelExecTimeoutEnabled ? "Si" : "No") << "\n";
    cout << "  -> Si hay límite de tiempo en la ejecución de kernels\n\n";

    // GPU integrada
    cout << "GPU integrada: " << (prop.integrated ? "Si" : "No") << "\n";
    cout << "  -> Si la GPU está integrada con la CPU\n\n";

    // Mapear memoria del host
    cout << "Puede mapear memoria del host: " << (prop.canMapHostMemory ? "Si" : "No") << "\n";
    cout << "  -> Si la GPU puede mapear memoria del host en su espacio de direcciones\n\n";

    // Modo de cómputo
    cout << "Modo de cómputo: " << prop.computeMode << "\n";
    cout << "  -> Modo en el que opera la GPU (0: Default, 1: Exclusive, 2: Prohibited)\n\n";

    // Máximo tamaño de textura 1D
    cout << "Tamaño máximo de textura 1D: " << prop.maxTexture1D << "\n";
    cout << "  -> Tamaño máximo de una textura unidimensional\n\n";

    // Máximo tamaño de textura 2D
    cout << "Tamaño máximo de textura 2D: (" << prop.maxTexture2D[0] << ", " << prop.maxTexture2D[1] << ")\n";
    cout << "  -> Tamaño máximo de una textura bidimensional\n\n";

    // Máximo tamaño de textura 3D
    cout << "Tamaño máximo de textura 3D: (" << prop.maxTexture3D[0] << ", " << prop.maxTexture3D[1] << ", " << prop.maxTexture3D[2] << ")\n";
    cout << "  -> Tamaño máximo de una textura tridimensional\n\n";

    // Kernels concurrentes
    cout << "Puede ejecutar kernels concurrentes: " << (prop.concurrentKernels ? "Si" : "No") << "\n";
    cout << "  -> Si la GPU puede ejecutar múltiples kernels al mismo tiempo\n\n";
}

int main() {

    // Contendrá el número total de GPUs en el sistema
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    // Recorre todas las GPUs disponibles en el sistema
    for (int device = 0; device < deviceCount; device++) {
        printDeviceProperties(device);
    }

    return 0;
}
