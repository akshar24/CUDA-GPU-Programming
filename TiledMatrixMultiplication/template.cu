#include <cuda_runtime.h> 
#include <device_launch_parameters.h> 
#include <wb.h>

#define TILE_WIDTH 5 //if needed, this value can be changed

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBColumns) {
  //@@ Insert code to implement tiled matrix multiplication here
  //@@ You have to use shared memory to write this kernel
    float ans = 0;
    __shared__ float dA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float dB[TILE_WIDTH][TILE_WIDTH];
    int rindex = threadIdx.y + blockIdx.y * TILE_WIDTH;
    int cindex = threadIdx.x + blockIdx.x * TILE_WIDTH;
    for (int i = 0; i < (TILE_WIDTH + numAColumns - 1) / TILE_WIDTH; i++) {
        int temp = i * TILE_WIDTH + threadIdx.x;
        if (temp < numAColumns && rindex < numARows) {
            dA[threadIdx.y][threadIdx.x] = A[rindex * numAColumns + i * TILE_WIDTH + threadIdx.x];
        }
        else {
            dA[threadIdx.y][threadIdx.x] = 0;
        }
        temp = i * TILE_WIDTH + threadIdx.y;
        if (temp < numAColumns && cindex < numBColumns) {
            dB[threadIdx.y][threadIdx.x] = B[cindex + numBColumns * (i * TILE_WIDTH + threadIdx.y)];

        }
        else {
            dB[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        for (int j = 0; j < TILE_WIDTH; ++j) {
            ans += dA[threadIdx.y][j] * dB[j][threadIdx.x];
        }
        __syncthreads();
    }
    if (rindex >= numARows || cindex >= numBColumns) {
        return;
    }
    else {
        C[((blockIdx.y * blockDim.y + threadIdx.y) * numBColumns) + (blockIdx.x * blockDim.x) + threadIdx.x] = ans;
    }
   
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  
  hostC = NULL;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  int sizea = numARows * numAColumns * sizeof(float);
  int sizeb = numBRows * numBColumns * sizeof(float);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  int sizec = numCRows * numCColumns * sizeof(float);

 
  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(sizec);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void**)&deviceA, sizea);
  cudaMalloc((void**)&deviceB, sizeb);
  cudaMalloc((void**)&deviceC, sizec);

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, sizea, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeb, cudaMemcpyHostToDevice);


  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
  dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<DimGrid,DimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, sizec, cudaMemcpyDeviceToHost);


  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
