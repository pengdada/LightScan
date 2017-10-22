
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
#include "cudaLib.cuh"
#include <stdio.h>

template<typename T>
__global__ void scanBrentKung(const T *input, T*output, int n)
{
	auto x = blockDim;
	auto y = gridDim;

	__shared__ T temp[1024*2];
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

extern "C" int mainBrentKung(int argc, char** argv) {
	int SIZE = 2048;
	std::vector<int> vecIn(SIZE), vecOut(SIZE);

	for (int i = 0; i < SIZE; i++) {
		vecIn[i] = i + 1;
	}

	DevData<int> devIn(SIZE), devOut(SIZE);
	devIn.CopyFromHost(&vecIn[0], vecIn.size(), vecIn.size(), 1);
	devOut.Zero();
	dim3 grids(1, 1, 1), blocks(SIZE/2, 1, 1);
	scanBrentKung<<<grids, blocks>>> (devIn.GetData(), devOut.GetData(), SIZE);
	devOut.CopyToHost(&vecOut[0], vecOut.size(), vecOut.size(), 1);	
	cudaDeviceSynchronize();

	//devOut.CopyToHost(&vecOut[0], 1, 1, 1);


	return 0;
}

