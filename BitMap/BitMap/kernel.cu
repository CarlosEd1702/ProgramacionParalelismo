#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include "bitmap_image.hpp"

using namespace std;

int main()
{
    /*string file_name("greenland_grid_velo.bmp");

    bitmap_image image(file_name);

    if (!image) {
        printf("test01() - Error - Failed to open '%s\n", file_name.c_str());
        return;
    }*/

    bitmap_image image(32, 32);

    for (size_t i = 0; i < 32; i++) {
        image.set_pixel(i, i, 23, 219, 230);
    }

    image.save_image("imagen_creada.bmp");

    return 0;
}
