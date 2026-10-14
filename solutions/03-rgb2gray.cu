
//
// CUDA program to convert an RGBA image to grayscale.
//
//

#include <iostream>
#include <vector>
#include <tuple>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

using namespace std;

//
// TODO #1
// Add the CUDA Kernel Device code to convert the RGB image to grayscale.
//
__global__ void convert(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, unsigned char *d_gray, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int index = y * width + x; // FIXED: correct indexing
        unsigned char r = d_r[index];
        unsigned char g = d_g[index];
        unsigned char b = d_b[index];
        d_gray[index] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

__host__ void executeKernel(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, unsigned char *d_gray, int width, int height, int threadsPerBlock)
{
    cout << "Executing kernel\n";

    // Calculate grid and block dimensions
    dim3 block(threadsPerBlock, threadsPerBlock, 1);
    dim3 grid(ceil((float) width / block.x), ceil((float) height / block.y), 1);

    cout << "threadsPerBlock = " << threadsPerBlock << " x " << threadsPerBlock << endl;
    cout << "grid = " << grid.x << " " << grid.y << ", " << "block = " << block.x << " " << block.y << endl;
    cout << "total pixels = " << width * height << ", total threads = " << grid.x * grid.y * block.x * block.y << endl;

    //
    // TODO #2
    // Call the CUDA kernel that converts the RGBA image to gray scale.
    //
    convert<<<grid, block>>>(d_r, d_g, d_b, d_gray, width, height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch convert kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    std::tuple<std::string, std::string, int> parsedCommandLineArgsTuple = parseCommandLineArguments(argc, argv);
    std::string inputImage = get<0>(parsedCommandLineArgsTuple);
    std::string outputImage = get<1>(parsedCommandLineArgsTuple);
    int threadsPerBlock = get<2>(parsedCommandLineArgsTuple);

    try
    {
        int width = 2048;
        int height = 2048;
        Image *img = read_raw_pixels(inputImage.c_str(), width, height);

        unsigned char *h_r = (unsigned char *)malloc(width * height * sizeof(unsigned char));
        unsigned char *h_g = (unsigned char *)malloc(width * height * sizeof(unsigned char));
        unsigned char *h_b = (unsigned char *)malloc(width * height * sizeof(unsigned char));

        for (int i = 0; i < width * height; ++i) {
            h_r[i] = img->data[i].r;
            h_g[i] = img->data[i].g;
            h_b[i] = img->data[i].b;
        }

        unsigned char *gray = (unsigned char *)malloc(width * height * sizeof(unsigned char));
        std::tuple<unsigned char *, unsigned char *, unsigned char *, unsigned char *> memoryTuple = allocateDeviceMemory(width, height);
        unsigned char *d_r = get<0>(memoryTuple);
        unsigned char *d_g = get<1>(memoryTuple);
        unsigned char *d_b = get<2>(memoryTuple);
        unsigned char *d_gray = get<3>(memoryTuple);

        copyFromHostToDevice(h_r, d_r, h_g, d_g, h_b, d_b, width, height);

        executeKernel(d_r, d_g, d_b, d_gray, width, height, threadsPerBlock);

        copyFromDeviceToHost(d_gray, gray, width, height);
        deallocateMemory(d_r, d_g, d_b, d_gray);
        cleanUpDevice();

        writeRAW(outputImage, gray, width, height);
    }
    catch (std::exception &error_)
    {
        cout << "Caught exception: " << error_.what() << endl;
        return 1;
    }
    return 0;
}
