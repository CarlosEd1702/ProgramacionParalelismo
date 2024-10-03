#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "bitmap_image.hpp"

using namespace std;

// Valores para N x M, se pueden cambiar aquí para definir en tiempo de compilación.
const int N = 64; // Ancho
const int M = 64; // Alto

void crearX(bitmap_image& image, int N, int M) {
    for (size_t i = 0; i < min(N, M); i++) {
        image.set_pixel(i, i, 23, 219, 230); // Diagonal de izquierda a derecha
        image.set_pixel(i, M - 1 - i, 23, 219, 230); // Diagonal de derecha a izquierda
    }
}

void crearCruz(bitmap_image& image, int N, int M) {
    // Línea vertical en el centro
    for (size_t i = 0; i < M; i++) {
        image.set_pixel(N / 2, i, 255, 0, 0); // Color rojo (R, G, B)
    }

    // Línea horizontal en el centro
    for (size_t i = 0; i < N; i++) {
        image.set_pixel(i, M / 2, 255, 0, 0); // Color rojo
    }
}

void crearCirculo(bitmap_image& image, int N, int M) {
    int center_x = N / 2; // Centro en el medio de la imagen
    int center_y = M / 2;
    int radius = min(N, M) / 3; // Radio del círculo ajustado para encajar dentro del tamaño

    // Dibujamos el círculo usando la ecuación (x - h)^2 + (y - k)^2 = r^2
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < M; y++) {
            int dx = x - center_x;
            int dy = y - center_y;
            if ((dx * dx + dy * dy) <= (radius * radius)) {
                image.set_pixel(x, y, 0, 255, 0); // Color verde
            }
        }
    }
}

int main() {
    int n = N; // Se puede modificar en tiempo de ejecución
    int m = M;

    // Si quieres cambiar el tamaño en tiempo de ejecución, se puede hacer así:
    cout << "Introduce el tamaño N (ancho) y M (alto): ";
    cin >> n >> m;

    // Crear imagen de la cruz
    bitmap_image image_cruz(n, m);
    crearCruz(image_cruz, n, m);
    image_cruz.save_image("imagen_cruz.bmp");

    // Crear imagen del círculo
    bitmap_image image_circulo(n, m);
    crearCirculo(image_circulo, n, m);
    image_circulo.save_image("imagen_circulo.bmp");

    // (Opcional) Crear imagen de la 'X' como ejemplo
    bitmap_image image_x(n, m);
    crearX(image_x, n, m);
    image_x.save_image("imagen_x.bmp");

    return 0;
}
