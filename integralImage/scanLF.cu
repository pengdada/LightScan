
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "cudaLib.cuh"
#include <stdio.h>

template<typename T>
__global__ void scanLF(const T *input, T*output, int n)
{
	auto x = blockDim;
	auto y = gridDim;

	unsigned int warpId, laneId;
	asm volatile("mov.u32 %0, %laneid;" : "=r"(laneId));
	warpId = threadIdx.x >> 5;
	assert(laneId == threadIdx.x % 32);

	T a, elem;

	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	elem = input[tid];

	for (int i = 1; i < 32; i <<= 1) {
		a = __shfl_up(elem, i);
	}

	//for (int i = 1; i <= 32; i <<= 1) {
	//	/*the first row of the matrix*/
	//	val = __shfl_up(elem[s], i);
	//	T va = val;
	//	if (laneId >= i) {
	//		elem[s] = op(elem[s], val);
	//	}
	//}






	__shared__ T temp[1024 * 2];
	int tdx = threadIdx.x; int offset = 1;
	temp[2 * tdx] = input[2 * tdx];
	temp[2 * tdx + 1] = input[2 * tdx + 1];

	for (int d = n >> 1; d > 0; d >>= 1)
	{
		__syncthreads();
		if (tdx < d)
		{
			int ai = offset*(2 * tdx + 1) - 1;
			int bi = offset*(2 * tdx + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (tdx == 0) temp[n - 1] = 0;
	for (int d = 1; d < n; d *= 2)
	{
		offset >>= 1; __syncthreads();
		if (tdx < d)
		{
			int ai = offset*(2 * tdx + 1) - 1;
			int bi = offset*(2 * tdx + 2) - 1;
			float t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();

	output[2 * tdx] = temp[2 * tdx];
	output[2 * tdx + 1] = temp[2 * tdx + 1];

}

int mainLF(int argc, char** argv) {
	int SIZE = 2048;
	std::vector<int> vecIn(SIZE), vecOut(SIZE);

	for (int i = 0; i < SIZE; i++) {
		vecIn[i] = i + 1;
	}

	DevData<int> devIn(SIZE), devOut(SIZE);
	devIn.CopyFromHost(&vecIn[0], vecIn.size(), vecIn.size(), 1);
	devOut.Zero();
	dim3 grids(1, 1, 1), blocks(SIZE / 2, 1, 1);
	scanLF<< <grids, blocks >> > (devIn.GetData(), devOut.GetData(), SIZE);
	devOut.CopyToHost(&vecOut[0], vecOut.size(), vecOut.size(), 1);
	cudaDeviceSynchronize();

	//devOut.CopyToHost(&vecOut[0], 1, 1, 1);


	return 0;
}

