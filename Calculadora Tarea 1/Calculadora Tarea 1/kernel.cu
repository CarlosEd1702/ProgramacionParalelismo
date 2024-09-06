#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// Función para la suma en la CPU
void add_cpu(int* a, int* b, int* c) {
    *c = *a + *b;
}

// Función para la suma en la GPU
__global__ void add_gpu(int* a, int* b, int* c) {
    *c = *a + *b;
}

// Función para la resta en la CPU
void subtract_cpu(int* a, int* b, int* c) {
    *c = *a - *b;
}

// Función para la resta en la GPU
__global__ void subtract_gpu(int* a, int* b, int* c) {
    *c = *a - *b;
}

// Función para multiplicar en la CPU
void multiply_cpu(int* a, int* b, int* c) {
    *c = *a * *b;
}

// Función para multiplicar en la GPU
__global__ void multiply_gpu(int* a, int* b, int* c) {
    *c = *a * *b;
}

// Función para dividir en la CPU
void divide_cpu(int* a, int* b, int* c) {
    *c = *a / *b;
}

// Función para dividir en la GPU
__global__ void divide_gpu(int* a, int* b, int* c) {
    *c = *a / *b;
}

int main() {
    // Variables en el Host (CPU)
    int operacion, a, b, c_cpu;
    char continuar;

    // Variables en el Device (GPU)
    int* d_a, * d_b, * d_c;
    int c_gpu;

    // Asignar memoria en la GPU para a, b y c
    cudaMalloc((void**)&d_a, sizeof(int));
    cudaMalloc((void**)&d_b, sizeof(int));
    cudaMalloc((void**)&d_c, sizeof(int));

    do {
        // Decidir cuál operación hacer
        cout << "Programacion Paralelismo - Tarea 1" << endl;
        cout << "\nQue operacion quieres hacer?" << endl;
        cout << "1. Suma\n2. Resta\n3. Multiplicacion\n4. Division\n" << endl;
        cin >> operacion;

        system("CLS");

        // Pedir los valores de entrada al usuario
        cout << "Ingresa el primer numero: ";
        cin >> a;
        cout << "\nIngresa el segundo numero: ";
        cin >> b;

        // Copiar los valores a y b del Host a la GPU
        cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

        switch (operacion) {
        case 1:
            // Suma en la CPU
            add_cpu(&a, &b, &c_cpu);
            cout << "Resultado de la suma en CPU: " << c_cpu << endl;
            cout << "Direccion de memoria del resultado en CPU: " << &c_cpu << endl;

            // Ejecutar la función de suma en la GPU
            add_gpu << <1, 1 >> > (d_a, d_b, d_c);

            // Copiar el resultado de la GPU de vuelta al Host
            cudaMemcpy(&c_gpu, d_c, sizeof(int), cudaMemcpyDeviceToHost);

            // Mostrar el resultado de la suma en GPU
            cout << "Resultado de la suma en GPU: " << c_gpu << endl;
            cout << "Direccion de memoria del resultado en GPU (en Host): " << &c_gpu << endl;
            cout << "Direccion de memoria del resultado en GPU (en Device): " << d_c << endl;
            break;

        case 2:
            // Resta en la CPU
            subtract_cpu(&a, &b, &c_cpu);
            cout << "Resultado de la resta en CPU: " << c_cpu << endl;
            cout << "Direccion de memoria del resultado en CPU: " << &c_cpu << endl;

            // Ejecutar la función de resta en la GPU
            subtract_gpu << <1, 1 >> > (d_a, d_b, d_c);

            // Copiar el resultado de la GPU de vuelta al Host
            cudaMemcpy(&c_gpu, d_c, sizeof(int), cudaMemcpyDeviceToHost);

            // Mostrar el resultado de la resta en GPU
            cout << "Resultado de la resta en GPU: " << c_gpu << endl;
            cout << "Dirección de memoria del resultado en GPU (en Host): " << &c_gpu << endl;
            cout << "Dirección de memoria del resultado en GPU (en Device): " << d_c << endl;
            break;

        case 3:
            // Multiplicacion en la CPU
            multiply_cpu(&a, &b, &c_cpu);
            cout << "Resultado de la multiplicacion en CPU: " << c_cpu << endl;
            cout << "Direccion de memoria del resultado en CPU: " << &c_cpu << endl;

            // Ejecutar la función de multiplicacion en la GPU
            multiply_gpu << <1, 1 >> > (d_a, d_b, d_c);

            // Copiar el resultado de la GPU de vuelta al Host
            cudaMemcpy(&c_gpu, d_c, sizeof(int), cudaMemcpyDeviceToHost);

            // Mostrar el resultado de la multiplicacion en GPU
            cout << "Resultado de la multiplicacion en GPU: " << c_gpu << endl;
            cout << "Dirección de memoria del resultado en GPU (en Host): " << &c_gpu << endl;
            cout << "Dirección de memoria del resultado en GPU (en Device): " << d_c << endl;
            break;

        case 4:
            // Division en la CPU
            divide_cpu(&a, &b, &c_cpu);
            cout << "Resultado de la division en CPU: " << c_cpu << endl;
            cout << "Dirección de memoria del resultado en CPU: " << &c_cpu << endl;

            // Ejecutar la función de división en la GPU
            divide_gpu << <1, 1 >> > (d_a, d_b, d_c);

            // Copiar el resultado de la GPU de vuelta al Host
            cudaMemcpy(&c_gpu, d_c, sizeof(int), cudaMemcpyDeviceToHost);

            // Mostrar el resultado de la division en GPU
            cout << "Resultado de la division en GPU: " << c_gpu << endl;
            cout << "Direccion de memoria del resultado en GPU (en Host): " << &c_gpu << endl;
            cout << "Direccion de memoria del resultado en GPU (en Device): " << d_c << endl;
            break;

        default:
            cout << "Operacion no valida!" << endl;
            break;
        }

        // Preguntar al usuario si quiere continuar
        cout << "\nDeseas realizar otra operacion? (S/N): ";
        cin >> continuar;

        system("CLS");  

    } while (continuar == 'S' || continuar == 's');

    // Liberar memoria en la GPU
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
