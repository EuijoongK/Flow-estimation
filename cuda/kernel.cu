#include "opticalflow.cuh"

#include <opencv2/core.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

void read_pixel8(cv::Mat* mat, unsigned char* data, int xSize, int ySize);
void write_pixel8(cv::Mat* mat, unsigned char* data, int xSize, int ySize);
void prepare_gaussian(float* buf, int filter_size, float std);

int main()
{
	FILE* fp;
	 // fp = fopen("test.raw", "wb");
	fopen_s(&fp, "test.raw", "wb");

	char* file_name = "C:/opencv/sources/samples/data/vtest.avi";
	cv::VideoCapture cap(file_name);

	if (!cap.isOpened()) {
		printf("Failed to open video file...\n");
		exit(-1);
	}

	cv::Mat frame1, frame2;

	if (!cap.read(frame1)) {
		printf("Failed to read video frame...\n");
		exit(-1);
	}

	int width = frame1.cols;
	int height = frame1.rows;
	cv::cvtColor(frame1, frame1, cv::COLOR_RGB2GRAY);
	int size = height * width;

	cv::VideoWriter writer;
	double frames_per_second = 20;
	writer.open("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), frames_per_second, cv::Size(width, height), true);
	
	unsigned char* data1, * data2;
	unsigned char* dev_data1, * dev_data2;
	int* diff_x, * diff_y, * diff_t;

	float* flow_x, * flow_y;
	float* dev_flow_x, * dev_flow_y;

	unsigned char* result;

	cudaMallocHost((void**)&data1, size);
	cudaMallocHost((void**)&data2, size);

	cudaMallocHost((void**)&result, size);
	cudaMalloc((void**)&dev_data1, size);
	cudaMalloc((void**)&dev_data2, size);
	cudaMalloc((void**)&diff_x, size * sizeof(int));
	cudaMalloc((void**)&diff_y, size * sizeof(int));
	cudaMalloc((void**)&diff_t, size * sizeof(int));

	cudaMallocHost((void**)&flow_x, sizeof(float) * width * height);
	cudaMallocHost((void**)&flow_y, sizeof(float) * width * height);

	cudaMalloc((void**)&dev_flow_x, sizeof(float) * width * height);
	cudaMalloc((void**)&dev_flow_y, sizeof(float) * width * height);

	int index = 0;
	read_pixel8(&frame1, data1, width, height);
	cudaMemcpy(dev_data1, data1, size, cudaMemcpyHostToDevice);

	dim3 threads(THREAD_X, THREAD_Y, 1);
	dim3 blocks((width + THREAD_X - 1) / THREAD_X, (height + THREAD_Y - 1) / THREAD_Y, 1);

	while (true) {
		printf("Frame %d..\n", index++);
		if (!cap.read(frame2)) {
			printf("End of video file...\n");
			break;
		}

		cv::cvtColor(frame2, frame2, cv::COLOR_RGB2GRAY);
		read_pixel8(&frame2, data2, width, height);
		cudaMemcpy(dev_data2, data2, size, cudaMemcpyHostToDevice);

		cv::Mat flow = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
		
		get_diff_x<<<blocks, threads>>>(dev_data1, diff_x, width, height);
		get_diff_y<<<blocks, threads>>>(dev_data1, diff_y, width, height);
		get_diff_t<<<blocks, threads>>>(dev_data1, dev_data2, diff_t, width, height);

		get_flow_vector<<<blocks, threads>>>(
			diff_x,
			diff_y,
			diff_t,
			dev_flow_x,
			dev_flow_y,
			width,
			height
		);
		
		cudaMemcpy(flow_x, dev_flow_x, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		cudaMemcpy(flow_y, dev_flow_y, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
		
		float threshold = 1000;
		for (int i = 0; i < height * width; ++i) {
			float val = flow_x[i] * flow_x[i] + flow_y[i] * flow_y[i];
			if (val < threshold) {
				val = 0;
			}
			*(result + i) = (unsigned char)val;
		}
		write_pixel8(&flow, result, width, height);

		/*for (int i = 0; i < height; i += FILTER) {
			for (int j = 0; j < width; j += FILTER) {
				float vx = *(flow_x + i * width + j);
				float vy = *(flow_y + i * width + j);

				float sz = vx * vx + vy * vy;

				if (sz > 5 && sz < 15) {
					cv::arrowedLine(flow, cv::Point2f(j, i), cv::Point2f(j + 5 * vy, i + 5 * vx), cv::Scalar(255, 255, 255), 1.5, 8, 0, 0.4);
				}
			}
		}*/

		cv::imshow("test", flow);
		writer.write(flow);
		cv::waitKey(1);

		cudaMemcpy(dev_data1, dev_data2, size, cudaMemcpyDeviceToDevice);
	}

	printf("Done!\n");
	cudaFreeHost(data1);
	cudaFreeHost(data2);
	cudaFreeHost(flow_x);
	cudaFreeHost(flow_y);
	cudaFreeHost(result);

	cudaFree(dev_data1);
	cudaFree(dev_data2);

	cudaFree(diff_x);
	cudaFree(diff_y);
	cudaFree(diff_t);

	cudaFree(dev_flow_x);
	cudaFree(dev_flow_y);

	cap.release();
	writer.release();
	fclose(fp);
}

void read_pixel8(cv::Mat* mat, unsigned char* data, int xSize, int ySize)
{
	for (int i = 0; i < ySize; ++i) {
		for (int j = 0; j < xSize; ++j) {
			*(data + i * xSize + j) = mat->at<unsigned char>(i, j);
		}
	}
}

void write_pixel8(cv::Mat* mat, unsigned char* data, int xSize, int ySize)
{
	for (int i = 0; i < ySize; ++i) {
		for (int j = 0; j < xSize; ++j) {
			mat->at<unsigned char>(i, j) = *(data + i * xSize + j);
		}
	}
}

void prepare_gaussian(float* buf, int filter_size, float std)
{

}