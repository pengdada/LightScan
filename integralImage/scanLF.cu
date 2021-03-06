
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "cudaLib.cuh"
#include <stdio.h>

__forceinline__ __device__ unsigned int getLaneid() {
	unsigned int laneId;
	asm volatile("mov.u32 %0, %laneid;" : "=r"(laneId));
	return laneId;
}

__forceinline__ __device__ unsigned int getWarpid() {
	return threadIdx.x >> 5;
}

template<typename T, int TYPE>
__global__ void Scan(const T *input, T*output, int n)
{
	//auto x = blockDim;
	//auto y = gridDim;

	unsigned int warpId = threadIdx.x >> 5;
	unsigned int laneId = threadIdx.x % 32;
	//asm volatile("mov.u32 %0, %laneid;" : "=r"(laneId));
	//warpId = threadIdx.x >> 5;
	//assert(laneId == threadIdx.x % 32);

	T a, elem = 0;

	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	if (tid < n)
		elem = input[tid];

	if (TYPE == 0){ //KS-pattern
#pragma unroll
		for (int i = 1; i <= 32; i <<= 1) {
			/*the first row of the matrix*/
			a = __shfl_up(elem, i);
			if (laneId >= i) {
				elem += a;
			}
		}
	}
	if (TYPE == 1) {//LF-scan
#pragma unroll
		for (int i = 1; i <= 32; i <<= 1) {
#pragma unroll
			for (int j = 1; j <= i; j++) {
				a = __shfl_up(elem, j);
				//if ((laneId % (i<<1)) == (i-1+j)) {
				if ((laneId & ((i << 1) - 1)) == (i - 1 + j)) {
					elem += a;
				}
			}
		}
	}
	if (TYPE == 2) { //KS-pattern
#pragma unroll
		for (int i = 1; i <= 32; i <<= 1) {
			a = __shfl(elem, i-1, i<<1);
			if ((laneId & ((i << 1) - 1)) >= i) {
				elem += a;
			}
		}
	}
	output[tid] = elem;

	//if (TYPE == 2) {
	//	//up-sweep
	//	for (int i = 1; i <= 32; i ++) {
	//		a = __shfl_up(elem, i);
	//		if ((i << 1) - 1 == (laneId & ((i << 1) - 1))) {
	//			elem += a;
	//		}
	//	}
	//	//down-sweep
	//}

	//if (tid < 1024) {
	//	output[tid] = elem;
	//	__shared__ int sMem[1024];
	//	sMem[tid] = elem;
	//	__syncthreads();

	//	a = elem;
	//}

	//for (int i = 1; i <= 32; i <<= 1) {
	//	/*the first row of the matrix*/
	//	val = __shfl_up(elem[s], i);
	//	T va = val;
	//	if (laneId >= i) {
	//		elem[s] = op(elem[s], val);
	//	}
	//}


	//__shared__ T temp[1024 * 2];
	//int tdx = threadIdx.x; int offset = 1;
	//temp[2 * tdx] = input[2 * tdx];
	//temp[2 * tdx + 1] = input[2 * tdx + 1];

	//for (int d = n >> 1; d > 0; d >>= 1)
	//{
	//	__syncthreads();
	//	if (tdx < d)
	//	{
	//		int ai = offset*(2 * tdx + 1) - 1;
	//		int bi = offset*(2 * tdx + 2) - 1;
	//		temp[bi] += temp[ai];
	//	}
	//	offset *= 2;
	//}
	//if (tdx == 0) temp[n - 1] = 0;
	//for (int d = 1; d < n; d *= 2)
	//{
	//	offset >>= 1; __syncthreads();
	//	if (tdx < d)
	//	{
	//		int ai = offset*(2 * tdx + 1) - 1;
	//		int bi = offset*(2 * tdx + 2) - 1;
	//		float t = temp[ai];
	//		temp[ai] = temp[bi];
	//		temp[bi] += t;
	//	}
	//}
	//__syncthreads();

	//output[2 * tdx] = temp[2 * tdx];
	//output[2 * tdx + 1] = temp[2 * tdx + 1];

}

int mainLF(int argc, char** argv) {
	int nType = atoi(argv[1]);
	printf("nType == %d\n", nType);
	int SIZE = 2048*100000;
	//int SIZE = 1024;
	std::vector<uint> vecIn(SIZE), vecOut(SIZE);

	for (int i = 0; i < SIZE; i++) {
		//vecIn[i] = i + 1;
		vecIn[i] = 1;
	}
	DevData<uint> devIn(SIZE), devOut(SIZE);
	devIn.CopyFromHost(&vecIn[0], vecIn.size(), vecIn.size(), 1);
	devOut.Zero();
	dim3 grids(SIZE/1024, 1, 1), blocks(1024, 1, 1);
	if (nType == 0)      Scan<uint, 0> <<<grids, blocks >>>(devIn.GetData(), devOut.GetData(), SIZE);
	else if (nType == 1) Scan<uint, 1><<<grids, blocks>>>(devIn.GetData(), devOut.GetData(), SIZE);
	else if (nType == 2) Scan<uint, 2> << <grids, blocks >> >(devIn.GetData(), devOut.GetData(), SIZE);
	devOut.CopyToHost(&vecOut[0], vecOut.size(), vecOut.size(), 1);
	cudaDeviceSynchronize();

	//devOut.CopyToHost(&vecOut[0], 1, 1, 1);

	return 0;
}

