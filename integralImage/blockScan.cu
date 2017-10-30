//#define __CUDA_ARCH__ 350
#include "cudaLib.cuh"
#include <stdio.h>
#include <vector>
#include <memory>

namespace BlockScan {
	template<typename T> __device__ __forceinline__
		void WarpPrefixSumLF(T& val, const uint& laneId, T& data) {
#pragma unroll
		for (int i = 1; i <= 32; i <<= 1) {
			val = __shfl(data, i - 1, i << 1);
			if ((laneId & ((i << 1) - 1)) >= i) {
				data += val;
			}
		}
	}

	static const uint WARP_SIZE = 32;
	template<typename T, uint BLOCK_SIZE, uint SMEM_COUNT>
	__global__ void blockScan(const T* __restrict__ dataIn, T* dataOut, uint width, uint widthStride, uint height, uint heightStride) {
		uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
		uint tidy = blockIdx.y * blockDim.y + threadIdx.y;
		uint warpId = threadIdx.x >> 5;
		uint laneId = threadIdx.x & 31;
		uint warpCount = blockDim.x >> 5;

		T data[BLOCK_SIZE], val;
		__shared__ T _smem[SMEM_COUNT][BLOCK_SIZE][WARP_SIZE + 1];
		__shared__ T smemSum[BLOCK_SIZE];

		auto smem = _smem[0];

		for (uint y = tidy*BLOCK_SIZE; y < height; y += gridDim.y*BLOCK_SIZE) {
			if (warpId == 0) {
				smemSum[laneId] = 0;
			}
			__syncthreads();
			for (uint x = tidx, cnt = 0; x < width; x += blockDim.x, cnt ++) {
				uint offset = y*widthStride + x;
				#pragma unroll
				for (int s = 0; s < BLOCK_SIZE; s++) {
					if (y + s < height) {
						data[s] = ldg(&dataIn[offset]);
						//data[s] = dataIn[offset];
						offset += widthStride;
						WarpPrefixSumLF(val, laneId, data[s]);
						if (laneId == WARP_SIZE - 1) smem[s][warpId] = data[s];
					}
				}
				__syncthreads();

				if (warpId == 0) {
					#pragma unroll
					for (int s = 0; s < BLOCK_SIZE; s++) {
						T a = smem[s][threadIdx.x];
						WarpPrefixSumLF(val, laneId, a);
						smem[s][threadIdx.x] = a;
					}
				}
				__syncthreads();
	
				if (warpId > 0) {
					#pragma unroll
					for (int s = 0; s < BLOCK_SIZE; s++) {
						data[s] += smem[s][warpId - 1];
					}
				}
				//__syncthreads();
				if (cnt > 0) {
					#pragma unroll
					for (int s = 0; s < BLOCK_SIZE; s++) {
						data[s] += smemSum[s];
					}
				}
				__syncthreads();
				if (threadIdx.x == blockDim.x - 1) {
					#pragma unroll
					for (int s = 0; s < BLOCK_SIZE; s++) {
						smemSum[s] = data[s];
					}
				}
				__syncthreads();
#if 0
#if 0
				offset = x*heightStride + y;
				#pragma unroll
				for (int s = 0; s < BLOCK_SIZE; s++) {
					dataOut[offset + s] = data[s];
				}
				//__syncthreads();
#else
				offset = y*widthStride + x;
				#pragma unroll
				for (int s = 0; s < BLOCK_SIZE; s++) {
					if (y + s < height) {
						dataOut[offset] = data[s];
						offset += widthStride;
					}
				}
#endif
#else
				for (int k = 0; k < warpCount; k += SMEM_COUNT) {
					if (warpId >= k && warpId < k + SMEM_COUNT) {
						auto csMem = _smem[warpId-k];
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
				uint _x = y & (~uint(31));
				uint _y = x & (~uint(31));
				offset = _y*heightStride + _x;
#pragma unroll
				for (int s = 0; s < BLOCK_SIZE; s++) {
					dataOut[offset+laneId] = data[s];
					offset += heightStride;
				}
				__syncthreads();
#endif
			}
		}
	}

	static void Test(int width, int height) {
		std::cout << "begin : TestBlockScan" << std::endl;
		float inc = 0;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		typedef float DataType;

		const uint BLOCK_SIZE = 32;
		//int width = 1024 * 2;
		//int height = 1024 * 2;
		int size = width*height;
		std::vector<DataType> vecA(size), vecB(size);
		//for (int i = 0; i < height-16; i += 32) std::fill(vecA.begin()+i*width, vecA.begin() + (i+16)*width, 1);

		std::fill(vecA.begin(), vecA.end(), 1);

		DevData<DataType> devA(width, height), devB(width, height), devTmp(height, width);
		devA.CopyFromHost(&vecA[0], width, width, height);

		DevStream SM;
		dim3 block_size(256 * 4, 1);
		dim3 grid_size1(1, UpDivide(height, BLOCK_SIZE));
		dim3 grid_size2(1, UpDivide(width, BLOCK_SIZE));
		float tm = 0;
		//tm = timeGetTime();
		cudaEventRecord(start, 0);
		BlockScan::blockScan<DataType, BLOCK_SIZE, 4 * sizeof(uint) / sizeof(DataType)> << <grid_size1, block_size, 0, SM.stream >> > (devA.GetData(), devTmp.GetData(), width, width, height, height);
		BlockScan::blockScan<DataType, BLOCK_SIZE, 4 * sizeof(uint) / sizeof(DataType)> << <grid_size2, block_size, 0, SM.stream >> > (devTmp.GetData(), devB.GetData(), height, height, width, width);
		cudaDeviceSynchronize();
		cudaEventRecord(stop, 0);
		//CUDA_CHECK_ERROR;


		//tm = timeGetTime() - tm;

		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&inc, start, stop);

		devB.CopyToHost(&vecB[0], width, width, height);

		FILE* fp = fopen("d:/int.raw", "wb");
		if(fp) {
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
		std::cout << "end : TestBlockScan" << std::endl;
	}
};



void TestBlockScan(){
//	BlockScan::Test(1 * 1024, 2 * 1024);
//	BlockScan::Test(2 * 1024, 1 * 1024);
	std::cout << "------------------------------------------------------" << std::endl;
	for(int i = 1; i < 10; i++){
		BlockScan::Test(i * 1024, i * 1024);
	}
	std::cout << "------------------------------------------------------" << std::endl;;
}