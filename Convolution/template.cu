#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <wb.h>

#define MASK_WIDTH 5
#define MAX_MASK_WIDTH 5
#define O_TILE_WIDTH 16
#define clamp(x) (min(max((x), 0.0), 1.0))
#define SHARED_MEM_SIZE (O_TILE_WIDTH + MAX_MASK_WIDTH  - 1)





//@@ INSERT CODE HERE 
//implement the tiled 2D convolution kernel with adjustments for channels
//use shared memory to reduce the number of global accesses, handle the boundary conditions when loading input list elements into the shared memory
//clamp your output values
__global__
void conv2D(float* input, float* out, const float* __restrict__ M, int channels, int width, int height) {
	__shared__ float N_smem[SHARED_MEM_SIZE][SHARED_MEM_SIZE];
	int n = MASK_WIDTH / 2;

	for (int channel = 0; channel < channels; ++channel) {
		int outIndex = threadIdx.y * O_TILE_WIDTH + threadIdx.x;
		
		int row_input = blockIdx.y * O_TILE_WIDTH + (outIndex / SHARED_MEM_SIZE) - n;
		int col_input = blockIdx.x * O_TILE_WIDTH + (outIndex % SHARED_MEM_SIZE) - n;
		if (row_input < height && row_input >= 0 && col_input >= 0 && col_input < width) {
			N_smem[outIndex / SHARED_MEM_SIZE][outIndex % SHARED_MEM_SIZE] = input[(row_input * width + col_input) * channels + channel];
		}
		else {
			N_smem[outIndex / SHARED_MEM_SIZE][outIndex % SHARED_MEM_SIZE] = 0;
		}
		outIndex += O_TILE_WIDTH * O_TILE_WIDTH;
		row_input = blockIdx.y * O_TILE_WIDTH + (outIndex / SHARED_MEM_SIZE) - n;
		col_input = blockIdx.x * O_TILE_WIDTH + (outIndex % SHARED_MEM_SIZE) - n;
		if ((outIndex / SHARED_MEM_SIZE) < SHARED_MEM_SIZE) {
			if (row_input < height && row_input >= 0 && col_input >= 0 && col_input < width) {
				N_smem[outIndex / SHARED_MEM_SIZE][outIndex % SHARED_MEM_SIZE] = input[(row_input * width + col_input) * channels + channel];
			}
			else {
				N_smem[outIndex / SHARED_MEM_SIZE][outIndex % SHARED_MEM_SIZE] = 0;
			}
		}
		__syncthreads();
		float pval = 0.0;
		for (int a = 0; a < MASK_WIDTH; ++a) {
			for (int b = 0; b < MASK_WIDTH; ++b) {
				pval += N_smem[threadIdx.y + a][threadIdx.x + b] * M[a * MASK_WIDTH + b];
			}
		}
		int row = blockIdx.y * O_TILE_WIDTH + threadIdx.y;
		int col = blockIdx.x * O_TILE_WIDTH + threadIdx.x;
		if (row < height && col < width) {
			out[(row * width + col) * channels + channel] = clamp(pval);
		}
		__syncthreads();
		
	}
}

int main(int argc, char *argv[]) {
  wbArg_t arg;
  int maskRows;
  int maskColumns;
  int imageChannels;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  char *inputMaskFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *hostMaskData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceMaskData;

  arg = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(arg, 0);
  inputMaskFile  = wbArg_getInputFile(arg, 1);

  inputImage   = wbImport(inputImageFile);
  hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

  assert(maskRows == MASK_WIDTH);    /* mask height is fixed to 5 */
  assert(maskColumns == MASK_WIDTH); /* mask width is fixed to 5 */

  imageWidth    = wbImage_getWidth(inputImage);
  imageHeight   = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);

  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

  hostInputImageData  = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ INSERT CODE HERE
  //allocate device memory
  int size = sizeof(float);
  int bytes = imageWidth * imageHeight * imageChannels * size;
  cudaMalloc((void**)&deviceInputImageData, bytes);
  cudaMalloc((void**)&deviceOutputImageData, bytes);
  cudaMalloc((void**)&deviceMaskData, maskRows * maskColumns * size);



  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ INSERT CODE HERE
  //copy host memory to device
  cudaMemcpy(deviceInputImageData, hostInputImageData, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMaskData, hostMaskData, maskRows * maskColumns * sizeof(float), cudaMemcpyHostToDevice);


  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ INSERT CODE HERE
  //initialize thread block and kernel grid dimensions
  //invoke CUDA kernel	
  dim3 DimGrid((imageWidth - 1) / O_TILE_WIDTH + 1, (imageHeight - 1) / O_TILE_WIDTH + 1, 1);
  dim3 DimBlock(O_TILE_WIDTH, O_TILE_WIDTH, 1);
  conv2D<<<DimGrid, DimBlock>>>(deviceInputImageData, deviceOutputImageData, deviceMaskData, imageChannels, imageWidth, imageHeight);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ INSERT CODE HERE
  //copy results from device to host	
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, bytes, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  wbSolution(arg, outputImage);

  //@@ INSERT CODE HERE
  //deallocate device memory	
  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceMaskData);

  free(hostMaskData);
  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}
