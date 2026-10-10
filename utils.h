#include <stdio.h>

typedef struct {
      unsigned char r;
      unsigned char g;
      unsigned char b;
      unsigned char a;
  } Pixel;

typedef struct {
      int width;
      int height;
      Pixel *data;
  } Image;

#include "rgb2gray.hpp"
#include <iostream>
#include <vector>
#include <tuple>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
using namespace std;


__host__ std::tuple<unsigned char *, unsigned char *, unsigned char *, unsigned char *> allocateDeviceMemory(int width, int height)
{
    cout << "Allocating GPU device memory\n";
    int num_image_pixels = width * height;
    size_t size = num_image_pixels * sizeof(unsigned char);

    // Allocate the device input vector d_r
    unsigned char *d_r = NULL;
    cudaError_t err = cudaMalloc(&d_r, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_r (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector d_g
    unsigned char *d_g = NULL;
    err = cudaMalloc(&d_g, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector d_b
    unsigned char *d_b = NULL;
    err = cudaMalloc(&d_b, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector d_gray
    unsigned char *d_gray = NULL;
    err = cudaMalloc(&d_gray, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_gray (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate device constant symbols for width and height
    cudaMemcpyToSymbol(d_width, &width, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_height, &height, sizeof(int), 0, cudaMemcpyHostToDevice);

    return {d_r, d_g, d_b, d_gray};
}


__host__ void copyFromHostToDevice(unsigned char *h_r, unsigned char *d_r, unsigned char *h_g, unsigned char *d_g, unsigned char *h_b, unsigned char *d_b, int width, int height)
{
    cout << "Copying from Host to Device\n";
    int num_image_pixels = width * height;
    size_t size = num_image_pixels * sizeof(unsigned char);

    cudaError_t err;
    err = cudaMemcpy(d_r, h_r, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector r from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_g, h_g, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector g from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector b from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


__host__ void copyFromDeviceToHost(unsigned char *d_gray, unsigned char *gray, int width, int height)
{
    cout << "Copying from Device to Host\n";
    // Copy the device result int array in device memory to the host result int array in host memory.
    size_t size = width * height * sizeof(unsigned char);

    cudaError_t err = cudaMemcpy(gray, d_gray, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy array d_gray from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Free device global memory
__host__ void deallocateMemory(unsigned char *d_r, unsigned char *d_g, unsigned char *d_b, unsigned char *d_gray)
{
    cout << "Deallocating GPU device memory\n";
    cudaError_t err = cudaFree(d_r);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_r (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_g);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_g (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_b);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector d_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_gray);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device int variable d_image_num_pixels (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Reset the device and exit
__host__ void cleanUpDevice()
{
    cout << "Cleaning CUDA device\n";
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaError_t err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__host__ std::tuple<std::string, std::string, int> parseCommandLineArguments(int argc, char *argv[])
{
    cout << "Parsing CLI arguments\n";
    int threadsPerBlock = 16;
    std::string inputImage = "pixels.raw";
    std::string outputImage = "grey.raw";

    for (int i = 1; i < argc; i++)
    {
        std::string option(argv[i]);
        i++;
        std::string value(argv[i]);
        if (option.compare("-i") == 0)
        {
            inputImage = value;
        }
        else if (option.compare("-o") == 0)
        {
            outputImage = value;
        }
        else if (option.compare("-t") == 0)
        {
            threadsPerBlock = atoi(value.c_str());
        }
    }

    cout << "inputImage: " << inputImage << " outputImage: " << outputImage << " threadsPerBlock dimension: " << threadsPerBlock << "\n";
    return {inputImage, outputImage, threadsPerBlock};
}



// Free image memory
void free_image(Image *img) {
      if (img) {
          free(img->data);
          free(img);
      }
  }

// Helper: Write grayscale RAW file
void writeRAW(std::string outputImage, unsigned char *gray, int width, int height){
    FILE *file = fopen(outputImage.c_str(), "wb");
    if (!file) {
        perror("Error opening output file");
        return;
    }

    size_t pixels_written = fwrite(gray, sizeof(unsigned char), width * height, file);
    if (pixels_written != width * height) {
        fprintf(stderr, "Error: Expected to write %d pixels, wrote %zu\n", width * height, pixels_written);
    }

    fclose(file);
    return;
}

// Helper: Read PNG file into RGB arrays
Image* read_raw_pixels(const char *filename, int width, int height)
{

  // Read raw pixel data from file
      FILE *file = fopen(filename, "rb");
      if (!file) {
          perror("Error opening file");
          return NULL;
      }

      // Allocate image structure
      Image *img = (Image*)malloc(sizeof(Image));
      if (!img) {
          fclose(file);
          return NULL;
      }

      // Allocate pixel array
      int num_pixels = width * height;
      img->data = (Pixel*)malloc(num_pixels * sizeof(Pixel));
      if (!img->data) {
          free(img);
          fclose(file);
          return NULL;
      }

      // Read pixel data
      size_t bytes_read = fread(img->data, sizeof(Pixel), num_pixels, file);
      if (bytes_read != num_pixels) {
          fprintf(stderr, "Error: Expected %d pixels, read %zu\n", num_pixels, bytes_read);
          free(img->data);
          free(img);
          fclose(file);
          return NULL;
      }

      fclose(file);
      return img;
}
