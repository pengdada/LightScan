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
	__global__ void blockScan(const T* dataIn, T* dataOut, T* dataWorking, uint width, uint widthStride, uint height, uint heightStride) {
		uint tidx = blockIdx.x * blockDim.x + threadIdx.x;
		uint tidy = blockIdx.y * blockDim.y + threadIdx.y;
		uint warpId = threadIdx.x >> 5;
		uint laneId = threadIdx.x & 31;
		uint warpCount = blockDim.x >> 5;

		uint data[BLOCK_SIZE], val;
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
						data[s] = dataIn[offset];
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
				uint _x = y >> 5 << 5;
				uint _y = x >> 5 << 5;
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

};

void TestBlockScan() {
	float inc = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	typedef uint DataType;

	const uint BLOCK_SIZE = 32;
	int width = 1024*1;
	int height = 1024*1;
	int size = width*height;
	std::vector<DataType> vecA(size), vecB(size);
	for (int i = 0; i < height-16; i += 16) {
		std::fill(vecA.begin()+i*width, vecA.begin() + (i+8)*width, 1);
	}

	DevData<DataType> devA(width, height), devB(width, height), devTmp(height, width);
	devA.CopyFromHost(&vecA[0], width, width, height);


	dim3 block_size(256*4, 1);
	dim3 grid_size(1, UpDivide(height, BLOCK_SIZE));

	float tm = timeGetTime();
	cudaEventRecord(start, 0);
	BlockScan::blockScan<uint, BLOCK_SIZE, 8*sizeof(DataType)/sizeof(uint)> << <grid_size, block_size >> > (devA.GetData(), devB.GetData(), devTmp.GetData(), width, width, height, height);
	cudaDeviceSynchronize();
	CUDA_CHECK_ERROR;

	cudaEventRecord(stop, 0);
	tm = timeGetTime() - tm;

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