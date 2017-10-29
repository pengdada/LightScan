#include "cudaLib.cuh"
#include <stdio.h>
#include <vector>
#include <memory>

namespace SerielScan {

	static const int WARP_SIZE = 32;
	static const int BLOCK_SIZE = WARP_SIZE;
	template<typename T, uint SMEM_COUNT>
	__global__ void serielScan(const T* dataIn, T* dataOut, int width, int widthStride, int height) {
		__shared__ T _smem[SMEM_COUNT][BLOCK_SIZE][WARP_SIZE + 1];
		__shared__ T smemSum[BLOCK_SIZE];
		auto smem = _smem[0];

		uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
		uint tidy = blockIdx.y * blockDim.y + threadIdx.y;
		uint warpId = threadIdx.x >> 5;
		uint laneId = threadIdx.x & 31;
		uint warpCount = blockDim.x >> 5;

		T data[BLOCK_SIZE];

		for (uint y = tidy*BLOCK_SIZE; y < height; y += gridDim.y*BLOCK_SIZE) {
			if (warpId == 0) {
				smemSum[laneId] = 0;
			}
			__syncthreads();
			for (uint x = tidx, cnt = 0; x < width; x += blockDim.x, cnt++) {
				uint offset = y*widthStride + x;
				#pragma unroll
				for (int s = 0; s < BLOCK_SIZE; s++) {
					if (y + s < height) {
						data[s] = ldg(&dataIn[offset]);
						offset += widthStride;
					}
				}
				T sum = data[0];
				#pragma unroll
				for (int s = 1; s < BLOCK_SIZE; s++) {
					sum += data[s];
					data[s] = sum;
				}
				__syncthreads();

				//rotate
				for (int k = 0; k < warpCount; k += SMEM_COUNT) {
					if (warpId >= k && warpId < k + SMEM_COUNT) {
						auto csMem = _smem[warpId - k];
						assert(warpId >= k);
						#pragma unroll
						for (int s = 0; s < BLOCK_SIZE; s++) {
							csMem[s][laneId] = data[s];
						}
						#pragma unroll
						for (int s = 0; s < BLOCK_SIZE; s++) {
							data[s] = csMem[laneId][s];
						}
					}
					__syncthreads();
				}

				#pragma unroll
				for (int s = 1; s < BLOCK_SIZE; s++) {
					data[s] += data[s - 1];
				}
				__syncthreads();
				//rotate
				for (int k = 0; k < warpCount; k += SMEM_COUNT) {
					if (warpId >= k && warpId < k + SMEM_COUNT) {
						auto csMem = _smem[warpId - k];
						assert(warpId >= k);
						#pragma unroll
						for (int s = 0; s < BLOCK_SIZE; s++) {
							csMem[s][laneId] = data[s];
						}
						#pragma unroll
						for (int s = 0; s < BLOCK_SIZE; s++) {
							data[s] = csMem[laneId][s];
						}
					}
					__syncthreads();
				}
				if (laneId == WARP_SIZE - 1) {
					#pragma unroll
					for (int s = 0; s < BLOCK_SIZE; s++) {
						smem[warpId][s] = data[s];
					}
					__syncthreads();
				}
				if (warpId == 0) {
					#pragma unroll
					for (int s = 1; s < BLOCK_SIZE; s++) {
						smem[laneId][s] += smem[laneId][s-1];
					}
					__syncthreads();
				}

				if (warpId > 0) {
					#pragma unroll
					for (int s = 0; s < BLOCK_SIZE; s++) {
						data[s] = smem[warpId - 1][s];
					}
					__syncthreads();
				}
			}
		}
	}
};

void TestSerielScan() {
	float inc = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	typedef uint DataType;

	const uint BLOCK_SIZE = 32;
	int width = 1024 * 1;
	int height = 1024 * 2;
	int size = width*height;
	std::vector<DataType> vecA(size), vecB(size);
	//for (int i = 0; i < height-16; i += 32) std::fill(vecA.begin()+i*width, vecA.begin() + (i+16)*width, 1);

	for (int i = 0; i < height; i++) std::fill(vecA.begin(), vecA.end(), 1);


	DevData<DataType> devA(width, height), devB(width, height), devTmp(height, width);
	devA.CopyFromHost(&vecA[0], width, width, height);

	DevStream SM;
	dim3 block_size(256 * 4, 1);
	dim3 grid_size1(1, UpDivide(height, BLOCK_SIZE));
	dim3 grid_size2(1, UpDivide(width, BLOCK_SIZE));
	float tm = 0;
	//tm = timeGetTime();
	cudaEventRecord(start, 0);
	//BlockScan::blockScan<uint, BLOCK_SIZE, 8 * sizeof(DataType) / sizeof(uint)> << <grid_size1, block_size, 0, SM.stream >> > (devA.GetData(), devTmp.GetData(), width, width, height, height);
	//BlockScan::blockScan<uint, BLOCK_SIZE, 8 * sizeof(DataType) / sizeof(uint)> << <grid_size2, block_size, 0, SM.stream >> > (devTmp.GetData(), devB.GetData(), height, height, width, width);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	//CUDA_CHECK_ERROR;


	//tm = timeGetTime() - tm;

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&inc, start, stop);

	devB.CopyToHost(&vecB[0], width, width, height);

	FILE* fp = fopen("d:/int.raw", "wb");
	if (fp) {
		fwrite(&vecB[0], sizeof(vecB[0]), width*height, fp);
		fclose(fp);
	}
	printf("%d, %d, total time = %f, %f\n", width, height, tm, inc);
	//cudaSyncDevice();
}

