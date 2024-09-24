#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define THREAD_X		32
#define THREAD_Y		32

#define FILTER			5
#define HFS				(FILTER - 1) / 2

#define ITERATION			20

__global__ void get_diff_x(unsigned char* input, int* output, int xSize, int ySize);
__global__ void get_diff_y(unsigned char* input, int* output, int xSize, int ySize);
__global__ void get_diff_t(unsigned char* input1, unsigned char* input2, int* output, int xSize, int ySize);

__global__ void gaussian8(unsigned char* input, unsigned char* output, int xSize, int ySize);

__global__ void get_flow_vector(
	int* diff_x,
	int* diff_y,
	int* diff_t,
	float* flow_x,
	float* flow_y,
	int xSize,
	int ySize
);