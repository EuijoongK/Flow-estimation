#include "opticalflow.cuh"

__global__ void get_diff_x(unsigned char* input, int* output, int xSize, int ySize)
{
	__shared__ int SM[THREAD_Y][THREAD_X + 2];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int ocx = blockDim.x * blockIdx.x;
	int ocy = blockDim.y * blockIdx.y;

	int x = ocx + tx;
	int y = ocy + ty;

	int x_offset[2];
	x_offset[0] = max(0, x - 1);
	x_offset[1] = min(xSize - 1, x + 1);

	SM[ty][tx + 1] = *(input + y * xSize + x);

	if (tx == 0) {
		SM[ty][tx] = *(input + y * xSize + x_offset[0]);
	}
	if (tx == THREAD_X - 1) {
		SM[ty][tx + 2] = *(input + y * xSize + x_offset[1]);
	}
	__syncthreads();

	*(output + y * xSize + x) = SM[ty][tx + 2] - SM[ty][tx];
}

__global__ void get_diff_y(unsigned char* input, int* output, int xSize, int ySize)
{
	__shared__ int SM[THREAD_Y + 2][THREAD_X];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int ocx = blockDim.x * blockIdx.x;
	int ocy = blockDim.y * blockIdx.y;

	int x = ocx + tx;
	int y = ocy + ty;

	int y_offset[2];
	y_offset[0] = max(0, y - 1);
	y_offset[1] = min(ySize - 1, y + 1);

	SM[ty + 1][tx] = *(input + y * xSize + x);

	if (ty == 0) {
		SM[ty][tx] = *(input + y_offset[0] * xSize + x);
	}
	if (ty == THREAD_Y - 1) {
		SM[ty + 2][tx] = *(input + y_offset[1] * xSize + x);
	}
	__syncthreads();

	* (output + y * xSize + x) = -SM[ty + 2][tx] + SM[ty][tx];
}

__global__ void get_diff_t(unsigned char* input1, unsigned char* input2, int* output, int xSize, int ySize)
{
	__shared__ int SM1[THREAD_Y][THREAD_X];
	__shared__ int SM2[THREAD_Y][THREAD_X];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int ocx = blockDim.x * blockIdx.x;
	int ocy = blockDim.y * blockIdx.y;

	int x = ocx + tx;
	int y = ocy + ty;

	SM1[ty][tx] = *(input1 + y * xSize + x);
	SM2[ty][tx] = *(input2 + y * xSize + x);
	__syncthreads();

	*(output + y * xSize + x) = SM2[ty][tx] - SM1[ty][tx];
}

__global__ void get_flow_vector(
	int* diff_x,
	int* diff_y,
	int* diff_t,
	float* flow_x,
	float* flow_y,
	int xSize,
	int ySize)
{
	__shared__ float DX[THREAD_Y + 2 * HFS][THREAD_X + 2 * HFS];
	__shared__ float DY[THREAD_Y + 2 * HFS][THREAD_X + 2 * HFS];
	__shared__ float DT[THREAD_Y + 2 * HFS][THREAD_X + 2 * HFS];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int ocx = blockDim.x * blockIdx.x;
	int ocy = blockDim.y * blockIdx.y;

	int x = ocx + tx;
	int y = ocy + ty;

	int x_offset[2], y_offset[2];
	x_offset[0] = max(0, x - HFS);
	x_offset[1] = min(xSize - 1, x + HFS);
	y_offset[0] = max(0, y - HFS);
	y_offset[1] = min(ySize - 1, y + HFS);

	DX[ty + HFS][tx + HFS] = (float)*(diff_x + y * xSize + x);
	DY[ty + HFS][tx + HFS] = (float)*(diff_y + y * xSize + x);
	DT[ty + HFS][tx + HFS] = (float)*(diff_t + y * xSize + x);

	int side_left = 0;
	int side_right = 0;
	
	// Filling left and right side of shared memory
	if (tx < HFS) {
		side_left = 1;
		DX[ty + HFS][tx] = (float)*(diff_x + y * xSize + x_offset[0]);
		DY[ty + HFS][tx] = (float)*(diff_y + y * xSize + x_offset[0]);
		DT[ty + HFS][tx] = (float)*(diff_t + y * xSize + x_offset[0]);
	}
	else if (tx + HFS >= blockDim.x) {
		side_right = 1;
		DX[ty + HFS][tx + 2 * HFS] = (float)*(diff_x + y * xSize + x_offset[1]);
		DY[ty + HFS][tx + 2 * HFS] = (float)*(diff_y + y * xSize + x_offset[1]);
		DT[ty + HFS][tx + 2 * HFS] = (float)*(diff_t + y * xSize + x_offset[1]);
	}

	// Filling top and bottom shared memory
	if (ty < HFS) {
		DX[ty][tx + HFS] = (float)*(diff_x + y_offset[0] * xSize + x);
		DY[ty][tx + HFS] = (float)*(diff_y + y_offset[0] * xSize + x);
		DT[ty][tx + HFS] = (float)*(diff_t + y_offset[0] * xSize + x);

		if (side_left) {
			DX[ty][tx] = (float)*(diff_x + y_offset[0] * xSize + x_offset[0]);
			DY[ty][tx] = (float)*(diff_y + y_offset[0] * xSize + x_offset[0]);
			DY[ty][tx] = (float)*(diff_y + y_offset[0] * xSize + x_offset[0]);
		}
		else if (side_right) {
			DX[ty][tx + 2 * HFS] = (float)*(diff_x + y_offset[0] * xSize + x_offset[1]);
			DY[ty][tx + 2 * HFS] = (float)*(diff_y + y_offset[0] * xSize + x_offset[1]);
			DT[ty][tx + 2 * HFS] = (float)*(diff_t + y_offset[0] * xSize + x_offset[1]);
		}
	}
	else if (ty + HFS >= blockDim.y) {
		DX[ty + 2 * HFS][tx + HFS] = (float)*(diff_x + y_offset[1] * xSize + x);
		DY[ty + 2 * HFS][tx + HFS] = (float)*(diff_y + y_offset[1] * xSize + x);
		DT[ty + 2 * HFS][tx + HFS] = (float)*(diff_t + y_offset[1] * xSize + x);

		if (side_left) {
			DX[ty + 2 * HFS][tx] = (float)*(diff_x + y_offset[1] * xSize + x_offset[0]);
			DY[ty + 2 * HFS][tx] = (float)*(diff_y + y_offset[1] * xSize + x_offset[0]);
			DT[ty + 2 * HFS][tx] = (float)*(diff_t + y_offset[1] * xSize + x_offset[0]);
		}
		else if (side_right) {
			DX[ty + 2 * HFS][tx + 2 * HFS] = (float)*(diff_x + y_offset[1] * xSize + x_offset[1]);
			DY[ty + 2 * HFS][tx + 2 * HFS] = (float)*(diff_y + y_offset[1] * xSize + x_offset[1]);
			DT[ty + 2 * HFS][tx + 2 * HFS] = (float)*(diff_t + y_offset[1] * xSize + x_offset[1]);
		}
	}

	__syncthreads();

	float M11 = 0;
	float M12 = 0;
	float M22 = 0;

	float b1 = 0;
	float b2 = 0;

	for (int i = 0; i < FILTER; ++i) {
		for (int j = 0; j < FILTER; ++j) {
			M11 += (DX[ty + i][tx + j] * DX[ty + i][tx + j]);
			M12 += (DX[ty + i][tx + j] * DY[ty + i][tx + j]);
			M22 += (DY[ty + i][tx + j] * DY[ty + i][tx + j]);

			b1 += (-DX[ty + i][tx + j] * DT[ty + i][tx + j]);
			b2 += (-DY[ty + i][tx + j] * DT[ty + i][tx + j]);
		}
	}

	float adbc = M11 * M22 - M12 * M12;
	float u = (M22 * b1 - M12 * b2) / adbc;
	float v = (-M12 * b1 + M22 * b2) / adbc;

	*(flow_x + y * xSize + x) = u;
	*(flow_y + y * xSize + x) = v;
}

__global__ void gaussian8(unsigned char* input, unsigned char* output, int xSize, int ySize) {
	__shared__ float SM[THREAD_Y + 2][THREAD_X + 2];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int ocx = blockDim.x * blockIdx.x;
	int ocy = blockDim.y * blockIdx.y;

	int x = ocx + tx;
	int y = ocy + ty;

	int x_offset[2], y_offset[2];
	x_offset[0] = max(0, x - HFS);
	x_offset[1] = min(xSize - 1, x + HFS);
	y_offset[0] = max(0, y - HFS);
	y_offset[1] = min(ySize - 1, y + HFS);

	SM[ty + HFS][tx + HFS] = (float)*(input + y * xSize + x);

	int side_left = 0;
	int side_right = 0;

	// Filling left and right side of shared memory
	if (tx < HFS) {
		side_left = 1;
		SM[ty + HFS][tx] = (float)*(input + y * xSize + x_offset[0]);
	}
	else if (tx + HFS >= blockDim.x) {
		side_right = 1;
		SM[ty + HFS][tx + 2 * HFS] = (float)*(input + y * xSize + x_offset[1]);
	}

	// Filling top and bottom shared memory
	if (ty < HFS) {
		SM[ty][tx + HFS] = (float)*(input + y_offset[0] * xSize + x);

		if (side_left) {
			SM[ty][tx] = (float)*(input + y_offset[0] * xSize + x_offset[0]);
		}
		else if (side_right) {
			SM[ty][tx + 2 * HFS] = (float)*(input + y_offset[0] * xSize + x_offset[1]);
		}
	}
	else if (ty + HFS >= blockDim.y) {
		SM[ty + 2 * HFS][tx + HFS] = (float)*(input + y_offset[1] * xSize + x);

		if (side_left) {
			SM[ty + 2 * HFS][tx] = (float)*(input + y_offset[1] * xSize + x_offset[0]);
		}
		else if (side_right) {
			SM[ty + 2 * HFS][tx + 2 * HFS] = (float)*(input + y_offset[1] * xSize + x_offset[1]);
		}
	}

	__syncthreads();


}