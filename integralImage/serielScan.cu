
#include "cudaLib.cuh"
#include <stdio.h>
#include <vector>
#include <memory>

namespace SerielScan {

	static const int WARP_SIZE = 32;
	static const int BLOCK_SIZE = WARP_SIZE;
	template<typename T, uint BLOCK_SIZE, uint SMEM_COUNT, uint BLOCK_DIM_X>
	__global__ void serielScan(const T* __restrict dataIn, T* dataOut, uint width, uint widthStride, uint height, uint heightStride) {
		__shared__ T _smem[SMEM_COUNT][BLOCK_SIZE][WARP_SIZE + 1];
		__shared__ T smemSum[BLOCK_SIZE];
		auto smem = _smem[0];

		uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
		uint tidy = blockIdx.y * blockDim.y + threadIdx.y;
		uint warpId = threadIdx.x >> 5;
		uint laneId = threadIdx.x & 31;
		uint warpCount = BLOCK_DIM_X >> 5;

		T data[BLOCK_SIZE];

		for (uint y = tidy*BLOCK_SIZE; y < height; y += gridDim.y*BLOCK_SIZE) {
			if (warpId == 0) {
				smemSum[laneId] = 0;
			}
			__syncthreads();
			for (uint x = tidx, cnt = 0; x < width; x += blockDim.x, cnt++) {
				uint offset = y*widthStride + x;
				#pragma unroll
				for (uint s = 0; s < BLOCK_SIZE; s++) {
					if (y + s < height) {
						data[s] = ldg(&dataIn[offset]);
						offset += widthStride;
					}
				}
				//rotate
				#pragma unroll
				for (int k = 0; k < warpCount; k += SMEM_COUNT) {
					if (warpId >= k && warpId < k + SMEM_COUNT) {
						auto csMem = _smem[warpId - k];
						assert(warpId >= k);
						#pragma unroll
						for (uint s = 0; s < BLOCK_SIZE; s++) {
							csMem[s][laneId] = data[s];
						}
						#pragma unroll
						for (uint s = 0; s < BLOCK_SIZE; s++) {
							data[s] = csMem[laneId][s];
						}
					}
					__syncthreads();
				}
				{
					T sum = data[0];
					#pragma unroll
					for (uint s = 1; s < BLOCK_SIZE; s++) {
						sum += data[s];
						data[s] = sum;
					}
					__syncthreads();
				}
				smem[warpId][laneId] = data[BLOCK_SIZE-1];
				__syncthreads();

				if (warpId == 0) {
					T sum = smem[0][laneId];
					#pragma unroll
					for (uint s = 1; s < BLOCK_SIZE; s++) {
						sum += smem[s][laneId];
						smem[s][laneId] = sum;
					}
				}
				__syncthreads();
				if (warpId > 0) {
					T sum = smem[warpId - 1][laneId];
					#pragma unroll
					for (uint s = 0; s < BLOCK_SIZE; s++) {
						data[s] += sum;
					}
				}
				__syncthreads();
				if (cnt > 0) {
					T sum = smemSum[laneId];
					#pragma unroll
					for (int s = 0; s < BLOCK_SIZE; s++) {
						data[s] += sum;
					}
				}
				__syncthreads();

				if (warpId == WARP_SIZE - 1) {
					smemSum[laneId] = data[BLOCK_SIZE - 1];
				}
				__syncthreads();

				uint _x = y & (~uint(31));
				uint _y = x & (~uint(31));
				offset = _y*heightStride + _x;
				#pragma unroll
				for (int s = 0; s < BLOCK_SIZE; s++) {
					dataOut[offset + laneId] = data[s];
					offset += heightStride;
				}
				__syncthreads();
			}
		}
	}
	void Test(int width, int height) {
	std::cout << "begin : TestSerielScan" << std::endl;
	float inc = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	typedef uint DataType;

	const uint BLOCK_SIZE = 32;
	const uint BLOCK_DIM_X = 256 * 4;
	//int width = 1024 * 2;
	//int height = 1024 * 2;
	int size = width*height;
	std::vector<DataType> vecA(size), vecB(size);
	//for (int i = 0; i < height-16; i += 32) std::fill(vecA.begin()+i*width, vecA.begin() + (i+16)*width, 1);

	std::fill(vecA.begin(), vecA.end(), 1);


	DevData<DataType> devA(width, height), devB(width, height), devTmp(height, width);
	devA.CopyFromHost(&vecA[0], width, width, height);

	DevStream SM;
	dim3 block_size(BLOCK_DIM_X, 1);
	dim3 grid_size1(1, UpDivide(height, BLOCK_SIZE));
	dim3 grid_size2(1, UpDivide(width, BLOCK_SIZE));
	float tm = 0;
	//tm = timeGetTime();
	cudaEventRecord(start, 0);
	SerielScan::serielScan<uint, BLOCK_SIZE, 8 * sizeof(DataType) / sizeof(uint), BLOCK_DIM_X> << <grid_size1, block_size, 0, SM.stream >> > (devA.GetData(), devTmp.GetData(), width, width, height, height);
	SerielScan::serielScan<uint, BLOCK_SIZE, 8 * sizeof(DataType) / sizeof(uint), BLOCK_DIM_X> << <grid_size2, block_size, 0, SM.stream >> > (devTmp.GetData(), devB.GetData(), height, height, width, width);
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	//CUDA_CHECK_ERROR;


	//tm = timeGetTime() - tm;

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&inc, start, stop);

	devB.CopyToHost(&vecB[0], width, width, height);

	FILE* fp = fopen("d:/ints.raw", "wb");
	if (fp) {
		fwrite(&vecB[0], sizeof(vecB[0]), width*height, fp);
		fclose(fp);
	}
	FILE* flog = fopen("d:/log.txt", "at");
	if(flog){
		fprintf(flog, "%f ", inc);
		fclose(flog);
	}
	printf("%d, %d, total time = %f, %f\n", width, height, tm, inc);
	//cudaSyncDevice();
	std::cout << "end : TestSerielScan" << std::endl;
}
};



void TestSerielScan(){
	std::cout << "------------------------------------------------------" << std::endl;
	//SerielScan::Test(1024, 1024);
	for(int i = 1; i < 10; i++){
		SerielScan::Test(i * 1024, i * 1024);
	}
	std::cout << "------------------------------------------------------" << std::endl;;
}