/*
 * Scan.cuh
 *
 *  Created on: Aug 25, 2015
 *      Author: Yongchao Liu
 *		Affiliation: Gerogia Institute of Technology
 *		Official Homepage: http://www.cc.gatech.edu/~yliu
 *		Personal Homepage: https://sites.google.com/site/yongchaosoftware
 */

#ifndef SCAN_CUH_
#define SCAN_CUH_
#include "Utils.cuh"
#pragma once

namespace Scan
{

//template<bool IsSharedMemory>
//struct WarpPrefixSum {
//	template<typename T>
//	static inline __device__ void Run(const int laneId, T& val, T* elem, int size) {
//		if (IsSharedMemory) {
//			T warp = elem[laneId];
//#pragma unroll
//			for (int i = 1; i <= 32; i *= 2) {
//				/*the first row of the matrix*/
//				val = __shfl_up(warp, i);
//				if (laneId >= i) {
//					warp = op(warp, val);
//				}
//			}
//			elem[laneId] = warp;
//		}
//		else {
//#pragma unroll
//			for (int i = 1; i <= 32; i *= 2) {
//				/*the first row of the matrix*/
//				val = __shfl_up(elem[i], i);
//				if (laneId >= i) {
//					elem[i] = op(elem[i], val);
//				}
//			}
//		}
//	}
//};
//



template<typename T, typename Sum, typename Comm, int ELEMENTS_PER_THREAD> inline
__device__ void priv_scan_stride_N(const int laneId, const int warpId, T* shrdMem,
	const T* __restrict dataIn, T* dataOut, Comm * partialSums,
	const unsigned int numBlocks) {
	int idx, gbid, base;
	T val, elem[ELEMENTS_PER_THREAD];
	Sum op;

	/*cyclic distribution of thread blocks*/
	for (gbid = blockIdx.x; gbid < numBlocks; gbid += gridDim.x) {

		/*get the base address*/
		base = ELEMENTS_PER_THREAD * (gbid * blockDim.x + warpId * 32);

		/*load 8 elements per thread*/
		idx = base + laneId;

#pragma unroll
		for (int s = 0; s < ELEMENTS_PER_THREAD; s++) {
			elem[s] = dataIn[idx];
			idx += 32;
		}

#pragma unroll
		for (int s = 0; s < ELEMENTS_PER_THREAD; s++) {
#pragma unroll
			for (int i = 1; i <= 32; i *= 2) {
				/*the first row of the matrix*/
				val = __shfl_up(elem[s], i);
				T va = val;
				if (laneId >= i) {
					elem[s] = op(elem[s], val);
				}
			}
		}
		/*perform intra-warp inclusive scan by broadcasting the last column of the matrix to each individual thread*/
#pragma unroll
		for (int s = 1; s < ELEMENTS_PER_THREAD; s++) {
			elem[s] = op(elem[s], __shfl(elem[s - 1], 31));
		}

		/*save its sum to shared memory for each warp in the thread block*/
		if (laneId == 31) {
			shrdMem[warpId] = elem[ELEMENTS_PER_THREAD-1];
		}
		__syncthreads();
		if (warpId == 0) {
			/*the share memory size is always equal to 32 * sizeof(T)*/
			T warp = shrdMem[laneId];
#pragma unroll
			for (int i = 1; i <= 32; i *= 2) {
				val = __shfl_up(warp, i);
				if (laneId >= i) {
					warp += val;
				}
			}
			/*save the prefix sums to shared memory*/
			shrdMem[laneId] = warp;
		}
		__syncthreads();

		/*update each element in the thread block*/
		val = 0;
		if (warpId > 0) {
			val = shrdMem[warpId - 1];
		}
#pragma unroll
		for (int s = 0; s < ELEMENTS_PER_THREAD; s++) {
			elem[s] = op(elem[s], val);
		}

		/*get the lead sum for the current thread block*/
		val = utils::busy_wait_comm<T, Comm>(partialSums, gbid, elem[ELEMENTS_PER_THREAD-1]);

#pragma unroll
		for (int s = 0; s < ELEMENTS_PER_THREAD; s++) {
			elem[s] = op(elem[s], val);
		}

		/*write the results to the output*/
		idx = base + laneId;
#pragma unroll
		for (int s = 0; s < ELEMENTS_PER_THREAD; s++) {
			dataOut[idx] = elem[s];
			idx += 32;
		}
	}
}



/*each thread stores 4 elements*/
template<typename T, typename Sum, typename Comm, int ELEMENTS_PER_THREAD> inline
__device__ void priv_scan_stride_4(const int laneId, const int warpId, T* shrdMem,
		const T* __restrict dataIn, T* dataOut, Comm * partialSums, const unsigned int numBlocks)
{
#if 1
	return priv_scan_stride_N<T, Sum, Comm, ELEMENTS_PER_THREAD>(laneId, warpId, shrdMem, dataIn, dataOut, partialSums, numBlocks);
#else
	int idx, gbid, base;
	T val, elem, elem2, elem3, elem4;
	Sum op;

	/*cyclic distribution of thread blocks*/
	for (gbid = blockIdx.x; gbid < numBlocks; gbid += gridDim.x) {

		/*get the base address*/
		base = ELEMENTS_PER_THREAD * (gbid * blockDim.x + warpId * 32);

		/*load 8 elements per thread*/
		idx = base + laneId;
		elem = dataIn[idx];
		idx += 32;
		elem2 = dataIn[idx];
		idx += 32;
		elem3 = dataIn[idx];
		idx += 32;
		elem4 = dataIn[idx];

		/*perform inclusive scan for each row of the matrix of size 8 by 32*/
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the first row of the matrix*/
			val = __shfl_up(elem, i);
			if (laneId >= i) {
				elem = op(elem, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the second row of the matrix*/
			val = __shfl_up(elem2, i);
			if (laneId >= i) {
				elem2 = op(elem2, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the third row of the matrix*/
			val = __shfl_up(elem3, i);
			if (laneId >= i) {
				elem3 = op(elem3, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fourth row of the matrix*/
			val = __shfl_up(elem4, i);
			if (laneId >= i) {
				elem4 = op(elem4, val);
			}
		}

		/*perform intra-warp inclusive scan by broadcasting the last column of the matrix to each individual thread*/
		elem2 = op(elem2, __shfl(elem, 31));
		elem3 = op(elem3, __shfl(elem2, 31));
		elem4 = op(elem4, __shfl(elem3, 31));

		/*save its sum to shared memory for each warp in the thread block*/
		if (laneId == 31) {
			shrdMem[warpId] = elem4;
		}
		__syncthreads();

		/*perform intra-block scan. Since the maximum number of threads per block is 1024 on CUDA-enabled GPUs, we can perform the prefix by
		 * a single warp*/
		if (warpId == 0) {
			/*the share memory size is always equal to 32 * sizeof(T)*/
			T warp = shrdMem[laneId];
#pragma unroll
			for (int i = 1; i <= 32; i *= 2) {
				val = __shfl_up(warp, i);
				if (laneId >= i) {
					warp += val;
				}
			}
			/*save the prefix sums to shared memory*/
			shrdMem[laneId] = warp;
		}
		__syncthreads();

		/*update each element in the thread block*/
		val = 0;
		if (warpId > 0) {
			val = shrdMem[warpId - 1];
		}
		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);

		/*get the lead sum for the current thread block*/
		val = utils::busy_wait_comm<T, Comm>(partialSums, gbid, elem4);

		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);

		/*write the results to the output*/
		idx = base + laneId;
		dataOut[idx] = elem;
		idx += 32;
		dataOut[idx] = elem2;
		idx += 32;
		dataOut[idx] = elem3;
		idx += 32;
		dataOut[idx] = elem4;
	}
#endif
}

/*each thread stores 8 elements*/
template<typename T, typename Sum, typename Comm, int ELEMENTS_PER_THREAD>
__device__ void priv_scan_stride_8(const int laneId, const int warpId, T* shrdMem,
		const T* __restrict dataIn, T* dataOut, Comm * partialSums,
		const unsigned int numBlocks) {
	int idx, gbid, base;
	T val, elem, elem2, elem3, elem4, elem5, elem6, elem7, elem8;
	Sum op;

	/*cyclic distribution of thread blocks*/
	for (gbid = blockIdx.x; gbid < numBlocks; gbid += gridDim.x) {

		/*get the base address*/
		base = ELEMENTS_PER_THREAD * (gbid * blockDim.x + warpId * 32);
		/*load 8 elements per thread*/
		idx = base + laneId;
		elem = dataIn[idx];
		idx += 32;
		elem2 = dataIn[idx];
		idx += 32;
		elem3 = dataIn[idx];
		idx += 32;
		elem4 = dataIn[idx];
		idx += 32;
		elem5 = dataIn[idx];
		idx += 32;
		elem6 = dataIn[idx];
		idx += 32;
		elem7 = dataIn[idx];
		idx += 32;
		elem8 = dataIn[idx];

		/*perform inclusive scan for each row of the matrix of size 8 by 32*/
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the first row of the matrix*/
			val = __shfl_up(elem, i);
			if (laneId >= i) {
				elem = op(elem, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the second row of the matrix*/
			val = __shfl_up(elem2, i);
			if (laneId >= i) {
				elem2 = op(elem2, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the third row of the matrix*/
			val = __shfl_up(elem3, i);
			if (laneId >= i) {
				elem3 = op(elem3, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fourth row of the matrix*/
			val = __shfl_up(elem4, i);
			if (laneId >= i) {
				elem4 = op(elem4, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fifth row of the matrix*/
			val = __shfl_up(elem5, i);
			if (laneId >= i) {
				elem5 = op(elem5, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the sixth row of the matrix*/
			val = __shfl_up(elem6, i);
			if (laneId >= i) {
				elem6 = op(elem6, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the seventh row of the matrix*/
			val = __shfl_up(elem7, i);
			if (laneId >= i) {
				elem7 = op(elem7, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the eighth row of the matrix*/
			val = __shfl_up(elem8, i);
			if (laneId >= i) {
				elem8 = op(elem8, val);
			}
		}
		/*perform intra-warp inclusive scan by broadcasting the last column of the matrix to each individual thread*/
		elem2 = op(elem2, __shfl(elem, 31));
		elem3 = op(elem3, __shfl(elem2, 31));
		elem4 = op(elem4, __shfl(elem3, 31));
		elem5 = op(elem5, __shfl(elem4, 31));
		elem6 = op(elem6, __shfl(elem5, 31));
		elem7 = op(elem7, __shfl(elem6, 31));
		elem8 = op(elem8, __shfl(elem7, 31));

		/*save its sum to shared memory for each warp in the thread block*/
		if (laneId == 31) {
			shrdMem[warpId] = elem8;
		}
		__syncthreads();

		/*perform intra-block scan*/
		if (warpId == 0) {
			T warp = shrdMem[laneId];
#pragma unroll
			for (int i = 1; i <= 32; i *= 2) {
				val = __shfl_up(warp, i);
				if (laneId >= i) {
					warp += val;
				}
			}
			/*save the prefix sums to shared memory*/
			shrdMem[laneId] = warp;
		}
		__syncthreads();

		/*update each element in the thread block*/
		val = warpId == 0 ? 0 : shrdMem[warpId - 1];
		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);

		/*get the lead sum for the current thread block*/
		val = utils::busy_wait_comm<T, Comm>(partialSums, gbid, elem8);

		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);

		/*write the results to the output*/
		idx = base + laneId;
		dataOut[idx] = elem;
		idx += 32;
		dataOut[idx] = elem2;
		idx += 32;
		dataOut[idx] = elem3;
		idx += 32;
		dataOut[idx] = elem4;
		idx += 32;
		dataOut[idx] = elem5;
		idx += 32;
		dataOut[idx] = elem6;
		idx += 32;
		dataOut[idx] = elem7;
		idx += 32;
		dataOut[idx] = elem8;
	}
}

template<typename T, typename Sum, typename Comm, int ELEMENTS_PER_THREAD>
__device__ void priv_scan_stride_16(const int laneId, const int warpId, T* shrdMem,
		const T* __restrict dataIn, T* dataOut, Comm* partialSums,
		const unsigned int numBlocks) {
	int idx, gbid, base;
	T val, elem, elem2, elem3, elem4, elem5, elem6, elem7, elem8;
	T elem9, elem10, elem11, elem12, elem13, elem14, elem15, elem16;
	Sum op;

	/*cyclic distribution of thread blocks*/
	for (gbid = blockIdx.x; gbid < numBlocks; gbid += gridDim.x) {

		/*get the base address*/
		base = ELEMENTS_PER_THREAD * (gbid * blockDim.x + warpId * 32);

		/*load 8 elements per thread*/
		idx = base + laneId;
		elem = dataIn[idx];
		idx += 32;
		elem2 = dataIn[idx];
		idx += 32;
		elem3 = dataIn[idx];
		idx += 32;
		elem4 = dataIn[idx];
		idx += 32;
		elem5 = dataIn[idx];
		idx += 32;
		elem6 = dataIn[idx];
		idx += 32;
		elem7 = dataIn[idx];
		idx += 32;
		elem8 = dataIn[idx];
		idx += 32;
		elem9 = dataIn[idx];
		idx += 32;
		elem10 = dataIn[idx];
		idx += 32;
		elem11 = dataIn[idx];
		idx += 32;
		elem12 = dataIn[idx];
		idx += 32;
		elem13 = dataIn[idx];
		idx += 32;
		elem14 = dataIn[idx];
		idx += 32;
		elem15 = dataIn[idx];
		idx += 32;
		elem16 = dataIn[idx];

		/*perform inclusive scan for each row of the matrix of size 8 by 32*/
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the first row of the matrix*/
			val = __shfl_up(elem, i);
			if (laneId >= i) {
				elem = op(elem, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the second row of the matrix*/
			val = __shfl_up(elem2, i);
			if (laneId >= i) {
				elem2 = op(elem2, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the third row of the matrix*/
			val = __shfl_up(elem3, i);
			if (laneId >= i) {
				elem3 = op(elem3, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fourth row of the matrix*/
			val = __shfl_up(elem4, i);
			if (laneId >= i) {
				elem4 = op(elem4, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fifth row of the matrix*/
			val = __shfl_up(elem5, i);
			if (laneId >= i) {
				elem5 = op(elem5, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the sixth row of the matrix*/
			val = __shfl_up(elem6, i);
			if (laneId >= i) {
				elem6 = op(elem6, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the seventh row of the matrix*/
			val = __shfl_up(elem7, i);
			if (laneId >= i) {
				elem7 = op(elem7, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the eighth row of the matrix*/
			val = __shfl_up(elem8, i);
			if (laneId >= i) {
				elem8 = op(elem8, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem9, i);
			if (laneId >= i) {
				elem9 = op(elem9, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem10, i);
			if (laneId >= i) {
				elem10 = op(elem10, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem11, i);
			if (laneId >= i) {
				elem11 = op(elem11, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem12, i);
			if (laneId >= i) {
				elem12 = op(elem12, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem13, i);
			if (laneId >= i) {
				elem13 = op(elem13, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem14, i);
			if (laneId >= i) {
				elem14 = op(elem14, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem15, i);
			if (laneId >= i) {
				elem15 = op(elem15, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem16, i);
			if (laneId >= i) {
				elem16 = op(elem16, val);
			}
		}
		/*perform intra-warp inclusive scan by broadcasting the last column of the matrix to each individual thread*/
		elem2 = op(elem2, __shfl(elem, 31));
		elem3 = op(elem3, __shfl(elem2, 31));
		elem4 = op(elem4, __shfl(elem3, 31));
		elem5 = op(elem5, __shfl(elem4, 31));
		elem6 = op(elem6, __shfl(elem5, 31));
		elem7 = op(elem7, __shfl(elem6, 31));
		elem8 = op(elem8, __shfl(elem7, 31));
		elem9 = op(elem9, __shfl(elem8, 31));
		elem10 = op(elem10, __shfl(elem9, 31));
		elem11 = op(elem11, __shfl(elem10, 31));
		elem12 = op(elem12, __shfl(elem11, 31));
		elem13 = op(elem13, __shfl(elem12, 31));
		elem14 = op(elem14, __shfl(elem13, 31));
		elem15 = op(elem15, __shfl(elem14, 31));
		elem16 = op(elem16, __shfl(elem15, 31));

		/*save its sum to shared memory for each warp in the thread block*/
		if (laneId == 31) {
			shrdMem[warpId] = elem16;
		}
		__syncthreads();

		/*perform intra-block scan*/
		if (warpId == 0) {
			T warp = shrdMem[laneId];
#pragma unroll
			for (int i = 1; i <= 32; i *= 2) {
				val = __shfl_up(warp, i);
				if (laneId >= i) {
					warp += val;
				}
			}
			/*save the prefix sums to shared memory*/
			shrdMem[laneId] = warp;
		}
		__syncthreads();

		/*update each element in the thread block*/
		val = warpId == 0 ? 0 : shrdMem[warpId - 1];
		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);

		/*get the lead sum for the current thread block*/
		val = utils::busy_wait_comm<T, Comm>(partialSums, gbid, elem16);

		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		/*write the results to the output*/
		idx = base + laneId;
		dataOut[idx] = elem;
		idx += 32;
		dataOut[idx] = elem2;
		idx += 32;
		dataOut[idx] = elem3;
		idx += 32;
		dataOut[idx] = elem4;
		idx += 32;
		dataOut[idx] = elem5;
		idx += 32;
		dataOut[idx] = elem6;
		idx += 32;
		dataOut[idx] = elem7;
		idx += 32;
		dataOut[idx] = elem8;
		idx += 32;
		dataOut[idx] = elem9;
		idx += 32;
		dataOut[idx] = elem10;
		idx += 32;
		dataOut[idx] = elem11;
		idx += 32;
		dataOut[idx] = elem12;
		idx += 32;
		dataOut[idx] = elem13;
		idx += 32;
		dataOut[idx] = elem14;
		idx += 32;
		dataOut[idx] = elem15;
		idx += 32;
		dataOut[idx] = elem16;
	}
}

template<typename T, typename Sum, typename Comm, int ELEMENTS_PER_THREAD>
__device__ void priv_scan_stride_20(const int laneId, const int warpId, T* shrdMem,
		const T* __restrict dataIn, T* dataOut, Comm* partialSums,
		const unsigned int numBlocks) {
	int idx, gbid, base;
	T val, elem, elem2, elem3, elem4, elem5, elem6, elem7, elem8;
	T elem9, elem10, elem11, elem12, elem13, elem14, elem15, elem16;
	T elem17, elem18, elem19, elem20;
	Sum op;

	/*cyclic distribution of thread blocks*/
	for (gbid = blockIdx.x; gbid < numBlocks; gbid += gridDim.x) {

		/*get the base address*/
		base = ELEMENTS_PER_THREAD * (gbid * blockDim.x + warpId * 32);

		/*load 8 elements per thread*/
		idx = base + laneId;
		elem = dataIn[idx];
		idx += 32;
		elem2 = dataIn[idx];
		idx += 32;
		elem3 = dataIn[idx];
		idx += 32;
		elem4 = dataIn[idx];
		idx += 32;
		elem5 = dataIn[idx];
		idx += 32;
		elem6 = dataIn[idx];
		idx += 32;
		elem7 = dataIn[idx];
		idx += 32;
		elem8 = dataIn[idx];
		idx += 32;
		elem9 = dataIn[idx];
		idx += 32;
		elem10 = dataIn[idx];
		idx += 32;
		elem11 = dataIn[idx];
		idx += 32;
		elem12 = dataIn[idx];
		idx += 32;
		elem13 = dataIn[idx];
		idx += 32;
		elem14 = dataIn[idx];
		idx += 32;
		elem15 = dataIn[idx];
		idx += 32;
		elem16 = dataIn[idx];
		idx += 32;
		elem17 = dataIn[idx];
		idx += 32;
		elem18 = dataIn[idx];
		idx += 32;
		elem19 = dataIn[idx];
		idx += 32;
		elem20 = dataIn[idx];

		/*perform inclusive scan for each row of the matrix of size 8 by 32*/
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the first row of the matrix*/
			val = __shfl_up(elem, i);
			if (laneId >= i) {
				elem = op(elem, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the second row of the matrix*/
			val = __shfl_up(elem2, i);
			if (laneId >= i) {
				elem2 = op(elem2, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the third row of the matrix*/
			val = __shfl_up(elem3, i);
			if (laneId >= i) {
				elem3 = op(elem3, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fourth row of the matrix*/
			val = __shfl_up(elem4, i);
			if (laneId >= i) {
				elem4 = op(elem4, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fifth row of the matrix*/
			val = __shfl_up(elem5, i);
			if (laneId >= i) {
				elem5 = op(elem5, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the sixth row of the matrix*/
			val = __shfl_up(elem6, i);
			if (laneId >= i) {
				elem6 = op(elem6, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the seventh row of the matrix*/
			val = __shfl_up(elem7, i);
			if (laneId >= i) {
				elem7 = op(elem7, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the eighth row of the matrix*/
			val = __shfl_up(elem8, i);
			if (laneId >= i) {
				elem8 = op(elem8, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem9, i);
			if (laneId >= i) {
				elem9 = op(elem9, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem10, i);
			if (laneId >= i) {
				elem10 = op(elem10, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem11, i);
			if (laneId >= i) {
				elem11 = op(elem11, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem12, i);
			if (laneId >= i) {
				elem12 = op(elem12, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem13, i);
			if (laneId >= i) {
				elem13 = op(elem13, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem14, i);
			if (laneId >= i) {
				elem14 = op(elem14, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem15, i);
			if (laneId >= i) {
				elem15 = op(elem15, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem16, i);
			if (laneId >= i) {
				elem16 = op(elem16, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem17, i);
			if (laneId >= i) {
				elem17 = op(elem17, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem18, i);
			if (laneId >= i) {
				elem18 = op(elem18, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem19, i);
			if (laneId >= i) {
				elem19 = op(elem19, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem20, i);
			if (laneId >= i) {
				elem20 = op(elem20, val);
			}
		}
		/*perform intra-warp inclusive scan by broadcasting the last column of the matrix to each individual thread*/
		elem2 = op(elem2, __shfl(elem, 31));
		elem3 = op(elem3, __shfl(elem2, 31));
		elem4 = op(elem4, __shfl(elem3, 31));
		elem5 = op(elem5, __shfl(elem4, 31));
		elem6 = op(elem6, __shfl(elem5, 31));
		elem7 = op(elem7, __shfl(elem6, 31));
		elem8 = op(elem8, __shfl(elem7, 31));
		elem9 = op(elem9, __shfl(elem8, 31));
		elem10 = op(elem10, __shfl(elem9, 31));
		elem11 = op(elem11, __shfl(elem10, 31));
		elem12 = op(elem12, __shfl(elem11, 31));
		elem13 = op(elem13, __shfl(elem12, 31));
		elem14 = op(elem14, __shfl(elem13, 31));
		elem15 = op(elem15, __shfl(elem14, 31));
		elem16 = op(elem16, __shfl(elem15, 31));
		elem17 = op(elem17, __shfl(elem16, 31));
		elem18 = op(elem18, __shfl(elem17, 31));
		elem19 = op(elem19, __shfl(elem18, 31));
		elem20 = op(elem20, __shfl(elem19, 31));

		/*save its sum to shared memory for each warp in the thread block*/
		if (laneId == 31) {
			shrdMem[warpId] = elem20;
		}
		__syncthreads();

		/*perform intra-block scan*/
		if (warpId == 0) {
			T warp = shrdMem[laneId];
#pragma unroll
			for (int i = 1; i <= 32; i *= 2) {
				val = __shfl_up(warp, i);
				if (laneId >= i) {
					warp += val;
				}
			}
			/*save the prefix sums to shared memory*/
			shrdMem[laneId] = warp;
		}
		__syncthreads();

		/*update each element in the thread block*/
		val = warpId == 0 ? 0 : shrdMem[warpId - 1];
		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);

		/*get the lead sum for the current thread block*/
		val = utils::busy_wait_comm<T, Comm>(partialSums, gbid, elem20);

		/*accumulate the sum for each block*/
		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);

		/*write the results to the output*/
		idx = base + laneId;
		dataOut[idx] = elem;
		idx += 32;
		dataOut[idx] = elem2;
		idx += 32;
		dataOut[idx] = elem3;
		idx += 32;
		dataOut[idx] = elem4;
		idx += 32;
		dataOut[idx] = elem5;
		idx += 32;
		dataOut[idx] = elem6;
		idx += 32;
		dataOut[idx] = elem7;
		idx += 32;
		dataOut[idx] = elem8;
		idx += 32;
		dataOut[idx] = elem9;
		idx += 32;
		dataOut[idx] = elem10;
		idx += 32;
		dataOut[idx] = elem11;
		idx += 32;
		dataOut[idx] = elem12;
		idx += 32;
		dataOut[idx] = elem13;
		idx += 32;
		dataOut[idx] = elem14;
		idx += 32;
		dataOut[idx] = elem15;
		idx += 32;
		dataOut[idx] = elem16;
		idx += 32;
		dataOut[idx] = elem17;
		idx += 32;
		dataOut[idx] = elem18;
		idx += 32;
		dataOut[idx] = elem19;
		idx += 32;
		dataOut[idx] = elem20;
	}
}

template<typename T, typename Sum, typename Comm, int ELEMENTS_PER_THREAD>
__device__ void priv_scan_stride_24(const int laneId, const int warpId, T* shrdMem,
		const T* __restrict dataIn, T* dataOut, Comm * partialSums,
		const unsigned int numBlocks) {
	int idx, gbid, base;
	T val, elem, elem2, elem3, elem4, elem5, elem6, elem7, elem8;
	T elem9, elem10, elem11, elem12, elem13, elem14, elem15, elem16;
	T elem17, elem18, elem19, elem20, elem21, elem22, elem23, elem24;
	Sum op;

	/*cyclic distribution of thread blocks*/
	for (gbid = blockIdx.x; gbid < numBlocks; gbid += gridDim.x) {

		/*get the base address*/
		base = ELEMENTS_PER_THREAD * (gbid * blockDim.x + warpId * 32);

		/*load 8 elements per thread*/
		idx = base + laneId;
		elem = dataIn[idx];
		idx += 32;
		elem2 = dataIn[idx];
		idx += 32;
		elem3 = dataIn[idx];
		idx += 32;
		elem4 = dataIn[idx];
		idx += 32;
		elem5 = dataIn[idx];
		idx += 32;
		elem6 = dataIn[idx];
		idx += 32;
		elem7 = dataIn[idx];
		idx += 32;
		elem8 = dataIn[idx];
		idx += 32;
		elem9 = dataIn[idx];
		idx += 32;
		elem10 = dataIn[idx];
		idx += 32;
		elem11 = dataIn[idx];
		idx += 32;
		elem12 = dataIn[idx];
		idx += 32;
		elem13 = dataIn[idx];
		idx += 32;
		elem14 = dataIn[idx];
		idx += 32;
		elem15 = dataIn[idx];
		idx += 32;
		elem16 = dataIn[idx];
		idx += 32;
		elem17 = dataIn[idx];
		idx += 32;
		elem18 = dataIn[idx];
		idx += 32;
		elem19 = dataIn[idx];
		idx += 32;
		elem20 = dataIn[idx];
		idx += 32;
		elem21 = dataIn[idx];
		idx += 32;
		elem22 = dataIn[idx];
		idx += 32;
		elem23 = dataIn[idx];
		idx += 32;
		elem24 = dataIn[idx];

		/*perform inclusive scan for each row of the matrix of size 8 by 32*/
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the first row of the matrix*/
			val = __shfl_up(elem, i);
			if (laneId >= i) {
				elem = op(elem, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the second row of the matrix*/
			val = __shfl_up(elem2, i);
			if (laneId >= i) {
				elem2 = op(elem2, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the third row of the matrix*/
			val = __shfl_up(elem3, i);
			if (laneId >= i) {
				elem3 = op(elem3, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fourth row of the matrix*/
			val = __shfl_up(elem4, i);
			if (laneId >= i) {
				elem4 = op(elem4, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fifth row of the matrix*/
			val = __shfl_up(elem5, i);
			if (laneId >= i) {
				elem5 = op(elem5, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the sixth row of the matrix*/
			val = __shfl_up(elem6, i);
			if (laneId >= i) {
				elem6 = op(elem6, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the seventh row of the matrix*/
			val = __shfl_up(elem7, i);
			if (laneId >= i) {
				elem7 = op(elem7, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the eighth row of the matrix*/
			val = __shfl_up(elem8, i);
			if (laneId >= i) {
				elem8 = op(elem8, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem9, i);
			if (laneId >= i) {
				elem9 = op(elem9, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem10, i);
			if (laneId >= i) {
				elem10 = op(elem10, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem11, i);
			if (laneId >= i) {
				elem11 = op(elem11, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem12, i);
			if (laneId >= i) {
				elem12 = op(elem12, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem13, i);
			if (laneId >= i) {
				elem13 = op(elem13, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem14, i);
			if (laneId >= i) {
				elem14 = op(elem14, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem15, i);
			if (laneId >= i) {
				elem15 = op(elem15, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem16, i);
			if (laneId >= i) {
				elem16 = op(elem16, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem17, i);
			if (laneId >= i) {
				elem17 = op(elem17, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem18, i);
			if (laneId >= i) {
				elem18 = op(elem18, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem19, i);
			if (laneId >= i) {
				elem19 = op(elem19, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem20, i);
			if (laneId >= i) {
				elem20 = op(elem20, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem21, i);
			if (laneId >= i) {
				elem21 = op(elem21, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem22, i);
			if (laneId >= i) {
				elem22 = op(elem22, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem23, i);
			if (laneId >= i) {
				elem23 = op(elem23, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem24, i);
			if (laneId >= i) {
				elem24 = op(elem24, val);
			}
		}

		/*perform intra-warp inclusive scan by broadcasting the last column of the matrix to each individual thread*/
		elem2 = op(elem2, __shfl(elem, 31));
		elem3 = op(elem3, __shfl(elem2, 31));
		elem4 = op(elem4, __shfl(elem3, 31));
		elem5 = op(elem5, __shfl(elem4, 31));
		elem6 = op(elem6, __shfl(elem5, 31));
		elem7 = op(elem7, __shfl(elem6, 31));
		elem8 = op(elem8, __shfl(elem7, 31));
		elem9 = op(elem9, __shfl(elem8, 31));
		elem10 = op(elem10, __shfl(elem9, 31));
		elem11 = op(elem11, __shfl(elem10, 31));
		elem12 = op(elem12, __shfl(elem11, 31));
		elem13 = op(elem13, __shfl(elem12, 31));
		elem14 = op(elem14, __shfl(elem13, 31));
		elem15 = op(elem15, __shfl(elem14, 31));
		elem16 = op(elem16, __shfl(elem15, 31));
		elem17 = op(elem17, __shfl(elem16, 31));
		elem18 = op(elem18, __shfl(elem17, 31));
		elem19 = op(elem19, __shfl(elem18, 31));
		elem20 = op(elem20, __shfl(elem19, 31));
		elem21 = op(elem21, __shfl(elem20, 31));
		elem22 = op(elem22, __shfl(elem21, 31));
		elem23 = op(elem23, __shfl(elem22, 31));
		elem24 = op(elem24, __shfl(elem23, 31));

		/*save its sum to shared memory for each warp in the thread block*/
		if (laneId == 31) {
			shrdMem[warpId] = elem24;
		}
		__syncthreads();

		/*perform intra-block scan*/
		if (warpId == 0) {
			T warp = shrdMem[laneId];
#pragma unroll
			for (int i = 1; i <= 32; i *= 2) {
				val = __shfl_up(warp, i);
				if (laneId >= i) {
					warp += val;
				}
			}
			/*save the prefix sums to shared memory*/
			shrdMem[laneId] = warp;
		}
		__syncthreads();

		/*update each element in the thread block*/
		val = warpId == 0 ? 0 : shrdMem[warpId - 1];
		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);

		/*get the lead sum for the current thread block*/
		val = utils::busy_wait_comm<T, Comm>(partialSums, gbid, elem24);

		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);

		/*write the results to the output*/
		idx = base + laneId;
		dataOut[idx] = elem;
		idx += 32;
		dataOut[idx] = elem2;
		idx += 32;
		dataOut[idx] = elem3;
		idx += 32;
		dataOut[idx] = elem4;
		idx += 32;
		dataOut[idx] = elem5;
		idx += 32;
		dataOut[idx] = elem6;
		idx += 32;
		dataOut[idx] = elem7;
		idx += 32;
		dataOut[idx] = elem8;
		idx += 32;
		dataOut[idx] = elem9;
		idx += 32;
		dataOut[idx] = elem10;
		idx += 32;
		dataOut[idx] = elem11;
		idx += 32;
		dataOut[idx] = elem12;
		idx += 32;
		dataOut[idx] = elem13;
		idx += 32;
		dataOut[idx] = elem14;
		idx += 32;
		dataOut[idx] = elem15;
		idx += 32;
		dataOut[idx] = elem16;
		idx += 32;
		dataOut[idx] = elem17;
		idx += 32;
		dataOut[idx] = elem18;
		idx += 32;
		dataOut[idx] = elem19;
		idx += 32;
		dataOut[idx] = elem20;
		idx += 32;
		dataOut[idx] = elem21;
		idx += 32;
		dataOut[idx] = elem22;
		idx += 32;
		dataOut[idx] = elem23;
		idx += 32;
		dataOut[idx] = elem24;
	}
}
template<typename T, typename Sum, typename Comm, int ELEMENTS_PER_THREAD>
__device__ void priv_scan_stride_28(const int laneId, const int warpId, T* shrdMem,
		const T* __restrict dataIn, T* dataOut, Comm* partialSums,
		const unsigned int numBlocks) {
	int idx, gbid, base;
	T val, elem, elem2, elem3, elem4, elem5, elem6, elem7, elem8;
	T elem9, elem10, elem11, elem12, elem13, elem14, elem15, elem16;
	T elem17, elem18, elem19, elem20, elem21, elem22, elem23, elem24;
	T elem25, elem26, elem27, elem28;
	Sum op;

	/*cyclic distribution of thread blocks*/
	for (gbid = blockIdx.x; gbid < numBlocks; gbid += gridDim.x) {

		/*get the base address*/
		base = ELEMENTS_PER_THREAD * (gbid * blockDim.x + warpId * 32);

		/*load 8 elements per thread*/
		idx = base + laneId;
		elem = dataIn[idx];
		idx += 32;
		elem2 = dataIn[idx];
		idx += 32;
		elem3 = dataIn[idx];
		idx += 32;
		elem4 = dataIn[idx];
		idx += 32;
		elem5 = dataIn[idx];
		idx += 32;
		elem6 = dataIn[idx];
		idx += 32;
		elem7 = dataIn[idx];
		idx += 32;
		elem8 = dataIn[idx];
		idx += 32;
		elem9 = dataIn[idx];
		idx += 32;
		elem10 = dataIn[idx];
		idx += 32;
		elem11 = dataIn[idx];
		idx += 32;
		elem12 = dataIn[idx];
		idx += 32;
		elem13 = dataIn[idx];
		idx += 32;
		elem14 = dataIn[idx];
		idx += 32;
		elem15 = dataIn[idx];
		idx += 32;
		elem16 = dataIn[idx];
		idx += 32;
		elem17 = dataIn[idx];
		idx += 32;
		elem18 = dataIn[idx];
		idx += 32;
		elem19 = dataIn[idx];
		idx += 32;
		elem20 = dataIn[idx];
		idx += 32;
		elem21 = dataIn[idx];
		idx += 32;
		elem22 = dataIn[idx];
		idx += 32;
		elem23 = dataIn[idx];
		idx += 32;
		elem24 = dataIn[idx];
		idx += 32;
		elem25 = dataIn[idx];
		idx += 32;
		elem26 = dataIn[idx];
		idx += 32;
		elem27 = dataIn[idx];
		idx += 32;
		elem28 = dataIn[idx];

		/*perform inclusive scan for each row of the matrix of size 8 by 32*/
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the first row of the matrix*/
			val = __shfl_up(elem, i);
			if (laneId >= i) {
				elem = op(elem, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the second row of the matrix*/
			val = __shfl_up(elem2, i);
			if (laneId >= i) {
				elem2 = op(elem2, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the third row of the matrix*/
			val = __shfl_up(elem3, i);
			if (laneId >= i) {
				elem3 = op(elem3, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fourth row of the matrix*/
			val = __shfl_up(elem4, i);
			if (laneId >= i) {
				elem4 = op(elem4, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fifth row of the matrix*/
			val = __shfl_up(elem5, i);
			if (laneId >= i) {
				elem5 = op(elem5, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the sixth row of the matrix*/
			val = __shfl_up(elem6, i);
			if (laneId >= i) {
				elem6 = op(elem6, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the seventh row of the matrix*/
			val = __shfl_up(elem7, i);
			if (laneId >= i) {
				elem7 = op(elem7, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the eighth row of the matrix*/
			val = __shfl_up(elem8, i);
			if (laneId >= i) {
				elem8 = op(elem8, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem9, i);
			if (laneId >= i) {
				elem9 = op(elem9, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem10, i);
			if (laneId >= i) {
				elem10 = op(elem10, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem11, i);
			if (laneId >= i) {
				elem11 = op(elem11, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem12, i);
			if (laneId >= i) {
				elem12 = op(elem12, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem13, i);
			if (laneId >= i) {
				elem13 = op(elem13, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem14, i);
			if (laneId >= i) {
				elem14 = op(elem14, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem15, i);
			if (laneId >= i) {
				elem15 = op(elem15, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem16, i);
			if (laneId >= i) {
				elem16 = op(elem16, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem17, i);
			if (laneId >= i) {
				elem17 = op(elem17, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem18, i);
			if (laneId >= i) {
				elem18 = op(elem18, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem19, i);
			if (laneId >= i) {
				elem19 = op(elem19, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem20, i);
			if (laneId >= i) {
				elem20 = op(elem20, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem21, i);
			if (laneId >= i) {
				elem21 = op(elem21, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem22, i);
			if (laneId >= i) {
				elem22 = op(elem22, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem23, i);
			if (laneId >= i) {
				elem23 = op(elem23, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem24, i);
			if (laneId >= i) {
				elem24 = op(elem24, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem25, i);
			if (laneId >= i) {
				elem25 = op(elem25, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem26, i);
			if (laneId >= i) {
				elem26 = op(elem26, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem27, i);
			if (laneId >= i) {
				elem27 = op(elem27, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem28, i);
			if (laneId >= i) {
				elem28 = op(elem28, val);
			}
		}
		/*perform intra-warp inclusive scan by broadcasting the last column of the matrix to each individual thread*/
		elem2 = op(elem2, __shfl(elem, 31));
		elem3 = op(elem3, __shfl(elem2, 31));
		elem4 = op(elem4, __shfl(elem3, 31));
		elem5 = op(elem5, __shfl(elem4, 31));
		elem6 = op(elem6, __shfl(elem5, 31));
		elem7 = op(elem7, __shfl(elem6, 31));
		elem8 = op(elem8, __shfl(elem7, 31));
		elem9 = op(elem9, __shfl(elem8, 31));
		elem10 = op(elem10, __shfl(elem9, 31));
		elem11 = op(elem11, __shfl(elem10, 31));
		elem12 = op(elem12, __shfl(elem11, 31));
		elem13 = op(elem13, __shfl(elem12, 31));
		elem14 = op(elem14, __shfl(elem13, 31));
		elem15 = op(elem15, __shfl(elem14, 31));
		elem16 = op(elem16, __shfl(elem15, 31));
		elem17 = op(elem17, __shfl(elem16, 31));
		elem18 = op(elem18, __shfl(elem17, 31));
		elem19 = op(elem19, __shfl(elem18, 31));
		elem20 = op(elem20, __shfl(elem19, 31));
		elem21 = op(elem21, __shfl(elem20, 31));
		elem22 = op(elem22, __shfl(elem21, 31));
		elem23 = op(elem23, __shfl(elem22, 31));
		elem24 = op(elem24, __shfl(elem23, 31));
		elem25 = op(elem25, __shfl(elem24, 31));
		elem26 = op(elem26, __shfl(elem25, 31));
		elem27 = op(elem27, __shfl(elem26, 31));
		elem28 = op(elem28, __shfl(elem27, 31));

		/*save its sum to shared memory for each warp in the thread block*/
		if (laneId == 31) {
			shrdMem[warpId] = elem28;
		}
		__syncthreads();

		/*perform intra-block scan*/
		if (warpId == 0) {
			T warp = shrdMem[laneId];
#pragma unroll
			for (int i = 1; i <= 32; i *= 2) {
				val = __shfl_up(warp, i);
				if (laneId >= i) {
					warp += val;
				}
			}
			/*save the prefix sums to shared memory*/
			shrdMem[laneId] = warp;
		}
		__syncthreads();

		/*update each element in the thread block*/
		val = warpId == 0 ? 0 : shrdMem[warpId - 1];
		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);
		elem25 = op(elem25, val);
		elem26 = op(elem26, val);
		elem27 = op(elem27, val);
		elem28 = op(elem28, val);

		/*get the lead sum for the current thread block*/
		val = utils::busy_wait_comm<T, Comm>(partialSums, gbid, elem28);

		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);
		elem25 = op(elem25, val);
		elem26 = op(elem26, val);
		elem27 = op(elem27, val);
		elem28 = op(elem28, val);

		/*write the results to the output*/
		idx = base + laneId;
		dataOut[idx] = elem;
		idx += 32;
		dataOut[idx] = elem2;
		idx += 32;
		dataOut[idx] = elem3;
		idx += 32;
		dataOut[idx] = elem4;
		idx += 32;
		dataOut[idx] = elem5;
		idx += 32;
		dataOut[idx] = elem6;
		idx += 32;
		dataOut[idx] = elem7;
		idx += 32;
		dataOut[idx] = elem8;
		idx += 32;
		dataOut[idx] = elem9;
		idx += 32;
		dataOut[idx] = elem10;
		idx += 32;
		dataOut[idx] = elem11;
		idx += 32;
		dataOut[idx] = elem12;
		idx += 32;
		dataOut[idx] = elem13;
		idx += 32;
		dataOut[idx] = elem14;
		idx += 32;
		dataOut[idx] = elem15;
		idx += 32;
		dataOut[idx] = elem16;
		idx += 32;
		dataOut[idx] = elem17;
		idx += 32;
		dataOut[idx] = elem18;
		idx += 32;
		dataOut[idx] = elem19;
		idx += 32;
		dataOut[idx] = elem20;
		idx += 32;
		dataOut[idx] = elem21;
		idx += 32;
		dataOut[idx] = elem22;
		idx += 32;
		dataOut[idx] = elem23;
		idx += 32;
		dataOut[idx] = elem24;
		idx += 32;
		dataOut[idx] = elem25;
		idx += 32;
		dataOut[idx] = elem26;
		idx += 32;
		dataOut[idx] = elem27;
		idx += 32;
		dataOut[idx] = elem28;
	}
}

template<typename T, typename Sum, typename Comm, int ELEMENTS_PER_THREAD>
__device__ void priv_scan_stride_32(const int laneId, const int warpId, T* shrdMem,
		const T* __restrict dataIn, T* dataOut, Comm* partialSums,
		const unsigned int numBlocks) {
	int idx, gbid, base;
	T val, elem, elem2, elem3, elem4, elem5, elem6, elem7, elem8;
	T elem9, elem10, elem11, elem12, elem13, elem14, elem15, elem16;
	T elem17, elem18, elem19, elem20, elem21, elem22, elem23, elem24;
	T elem25, elem26, elem27, elem28, elem29, elem30, elem31, elem32;
	Sum op;

	/*cyclic distribution of thread blocks*/
	for (gbid = blockIdx.x; gbid < numBlocks; gbid += gridDim.x) {

		/*get the base address*/
		base = ELEMENTS_PER_THREAD * (gbid * blockDim.x + warpId * 32);

		/*load 8 elements per thread*/
		idx = base + laneId;
		elem = dataIn[idx];
		idx += 32;
		elem2 = dataIn[idx];
		idx += 32;
		elem3 = dataIn[idx];
		idx += 32;
		elem4 = dataIn[idx];
		idx += 32;
		elem5 = dataIn[idx];
		idx += 32;
		elem6 = dataIn[idx];
		idx += 32;
		elem7 = dataIn[idx];
		idx += 32;
		elem8 = dataIn[idx];
		idx += 32;
		elem9 = dataIn[idx];
		idx += 32;
		elem10 = dataIn[idx];
		idx += 32;
		elem11 = dataIn[idx];
		idx += 32;
		elem12 = dataIn[idx];
		idx += 32;
		elem13 = dataIn[idx];
		idx += 32;
		elem14 = dataIn[idx];
		idx += 32;
		elem15 = dataIn[idx];
		idx += 32;
		elem16 = dataIn[idx];
		idx += 32;
		elem17 = dataIn[idx];
		idx += 32;
		elem18 = dataIn[idx];
		idx += 32;
		elem19 = dataIn[idx];
		idx += 32;
		elem20 = dataIn[idx];
		idx += 32;
		elem21 = dataIn[idx];
		idx += 32;
		elem22 = dataIn[idx];
		idx += 32;
		elem23 = dataIn[idx];
		idx += 32;
		elem24 = dataIn[idx];
		idx += 32;
		elem25 = dataIn[idx];
		idx += 32;
		elem26 = dataIn[idx];
		idx += 32;
		elem27 = dataIn[idx];
		idx += 32;
		elem28 = dataIn[idx];
		idx += 32;
		elem29 = dataIn[idx];
		idx += 32;
		elem30 = dataIn[idx];
		idx += 32;
		elem31 = dataIn[idx];
		idx += 32;
		elem32 = dataIn[idx];

		/*perform inclusive scan for each row of the matrix of size 8 by 32*/
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the first row of the matrix*/
			val = __shfl_up(elem, i);
			if (laneId >= i) {
				elem = op(elem, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the second row of the matrix*/
			val = __shfl_up(elem2, i);
			if (laneId >= i) {
				elem2 = op(elem2, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the third row of the matrix*/
			val = __shfl_up(elem3, i);
			if (laneId >= i) {
				elem3 = op(elem3, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fourth row of the matrix*/
			val = __shfl_up(elem4, i);
			if (laneId >= i) {
				elem4 = op(elem4, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fifth row of the matrix*/
			val = __shfl_up(elem5, i);
			if (laneId >= i) {
				elem5 = op(elem5, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the sixth row of the matrix*/
			val = __shfl_up(elem6, i);
			if (laneId >= i) {
				elem6 = op(elem6, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the seventh row of the matrix*/
			val = __shfl_up(elem7, i);
			if (laneId >= i) {
				elem7 = op(elem7, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the eighth row of the matrix*/
			val = __shfl_up(elem8, i);
			if (laneId >= i) {
				elem8 = op(elem8, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem9, i);
			if (laneId >= i) {
				elem9 = op(elem9, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem10, i);
			if (laneId >= i) {
				elem10 = op(elem10, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem11, i);
			if (laneId >= i) {
				elem11 = op(elem11, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem12, i);
			if (laneId >= i) {
				elem12 = op(elem12, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem13, i);
			if (laneId >= i) {
				elem13 = op(elem13, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem14, i);
			if (laneId >= i) {
				elem14 = op(elem14, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem15, i);
			if (laneId >= i) {
				elem15 = op(elem15, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem16, i);
			if (laneId >= i) {
				elem16 = op(elem16, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem17, i);
			if (laneId >= i) {
				elem17 = op(elem17, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem18, i);
			if (laneId >= i) {
				elem18 = op(elem18, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem19, i);
			if (laneId >= i) {
				elem19 = op(elem19, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem20, i);
			if (laneId >= i) {
				elem20 = op(elem20, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem21, i);
			if (laneId >= i) {
				elem21 = op(elem21, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem22, i);
			if (laneId >= i) {
				elem22 = op(elem22, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem23, i);
			if (laneId >= i) {
				elem23 = op(elem23, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem24, i);
			if (laneId >= i) {
				elem24 = op(elem24, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem25, i);
			if (laneId >= i) {
				elem25 = op(elem25, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem26, i);
			if (laneId >= i) {
				elem26 = op(elem26, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem27, i);
			if (laneId >= i) {
				elem27 = op(elem27, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem28, i);
			if (laneId >= i) {
				elem28 = op(elem28, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem29, i);
			if (laneId >= i) {
				elem29 = op(elem29, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem30, i);
			if (laneId >= i) {
				elem30 = op(elem30, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem31, i);
			if (laneId >= i) {
				elem31 = op(elem31, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem32, i);
			if (laneId >= i) {
				elem32 = op(elem32, val);
			}
		}

		/*perform intra-warp inclusive scan by broadcasting the last column of the matrix to each individual thread*/
		elem2 = op(elem2, __shfl(elem, 31));
		elem3 = op(elem3, __shfl(elem2, 31));
		elem4 = op(elem4, __shfl(elem3, 31));
		elem5 = op(elem5, __shfl(elem4, 31));
		elem6 = op(elem6, __shfl(elem5, 31));
		elem7 = op(elem7, __shfl(elem6, 31));
		elem8 = op(elem8, __shfl(elem7, 31));
		elem9 = op(elem9, __shfl(elem8, 31));
		elem10 = op(elem10, __shfl(elem9, 31));
		elem11 = op(elem11, __shfl(elem10, 31));
		elem12 = op(elem12, __shfl(elem11, 31));
		elem13 = op(elem13, __shfl(elem12, 31));
		elem14 = op(elem14, __shfl(elem13, 31));
		elem15 = op(elem15, __shfl(elem14, 31));
		elem16 = op(elem16, __shfl(elem15, 31));
		elem17 = op(elem17, __shfl(elem16, 31));
		elem18 = op(elem18, __shfl(elem17, 31));
		elem19 = op(elem19, __shfl(elem18, 31));
		elem20 = op(elem20, __shfl(elem19, 31));
		elem21 = op(elem21, __shfl(elem20, 31));
		elem22 = op(elem22, __shfl(elem21, 31));
		elem23 = op(elem23, __shfl(elem22, 31));
		elem24 = op(elem24, __shfl(elem23, 31));
		elem25 = op(elem25, __shfl(elem24, 31));
		elem26 = op(elem26, __shfl(elem25, 31));
		elem27 = op(elem27, __shfl(elem26, 31));
		elem28 = op(elem28, __shfl(elem27, 31));
		elem29 = op(elem29, __shfl(elem28, 31));
		elem30 = op(elem30, __shfl(elem29, 31));
		elem31 = op(elem31, __shfl(elem30, 31));
		elem32 = op(elem32, __shfl(elem31, 31));

		/*save its sum to shared memory for each warp in the thread block*/
		if (laneId == 31) {
			shrdMem[warpId] = elem32;
		}
		__syncthreads();

		/*perform intra-block scan*/
		if (warpId == 0) {
			T warp = shrdMem[laneId];
#pragma unroll
			for (int i = 1; i <= 32; i *= 2) {
				val = __shfl_up(warp, i);
				if (laneId >= i) {
					warp += val;
				}
			}
			/*save the prefix sums to shared memory*/
			shrdMem[laneId] = warp;
		}
		__syncthreads();

		/*update each element in the thread block*/
		val = warpId == 0 ? 0 : shrdMem[warpId - 1];
		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);
		elem25 = op(elem25, val);
		elem26 = op(elem26, val);
		elem27 = op(elem27, val);
		elem28 = op(elem28, val);
		elem29 = op(elem29, val);
		elem30 = op(elem30, val);
		elem31 = op(elem31, val);
		elem32 = op(elem32, val);

		/*get the lead sum for the current thread block*/
		val = utils::busy_wait_comm<T, Comm>(partialSums, gbid, elem32);

		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);
		elem25 = op(elem25, val);
		elem26 = op(elem26, val);
		elem27 = op(elem27, val);
		elem28 = op(elem28, val);
		elem29 = op(elem29, val);
		elem30 = op(elem30, val);
		elem31 = op(elem31, val);
		elem32 = op(elem32, val);

		/*write the results to the output*/
		idx = base + laneId;
		dataOut[idx] = elem;
		idx += 32;
		dataOut[idx] = elem2;
		idx += 32;
		dataOut[idx] = elem3;
		idx += 32;
		dataOut[idx] = elem4;
		idx += 32;
		dataOut[idx] = elem5;
		idx += 32;
		dataOut[idx] = elem6;
		idx += 32;
		dataOut[idx] = elem7;
		idx += 32;
		dataOut[idx] = elem8;
		idx += 32;
		dataOut[idx] = elem9;
		idx += 32;
		dataOut[idx] = elem10;
		idx += 32;
		dataOut[idx] = elem11;
		idx += 32;
		dataOut[idx] = elem12;
		idx += 32;
		dataOut[idx] = elem13;
		idx += 32;
		dataOut[idx] = elem14;
		idx += 32;
		dataOut[idx] = elem15;
		idx += 32;
		dataOut[idx] = elem16;
		idx += 32;
		dataOut[idx] = elem17;
		idx += 32;
		dataOut[idx] = elem18;
		idx += 32;
		dataOut[idx] = elem19;
		idx += 32;
		dataOut[idx] = elem20;
		idx += 32;
		dataOut[idx] = elem21;
		idx += 32;
		dataOut[idx] = elem22;
		idx += 32;
		dataOut[idx] = elem23;
		idx += 32;
		dataOut[idx] = elem24;
		idx += 32;
		dataOut[idx] = elem25;
		idx += 32;
		dataOut[idx] = elem26;
		idx += 32;
		dataOut[idx] = elem27;
		idx += 32;
		dataOut[idx] = elem28;
		idx += 32;
		dataOut[idx] = elem29;
		idx += 32;
		dataOut[idx] = elem30;
		idx += 32;
		dataOut[idx] = elem31;
		idx += 32;
		dataOut[idx] = elem32;
	}
}

template<typename T, typename Sum, typename Comm, int ELEMENTS_PER_THREAD>
__device__ void priv_scan_stride_36(const int laneId, const int warpId, T* shrdMem,
		const T* __restrict dataIn, T* dataOut, Comm* partialSums,
		const unsigned int numBlocks) {
	int idx, gbid, base;
	T val, elem, elem2, elem3, elem4, elem5, elem6, elem7, elem8;
	T elem9, elem10, elem11, elem12, elem13, elem14, elem15, elem16;
	T elem17, elem18, elem19, elem20, elem21, elem22, elem23, elem24;
	T elem25, elem26, elem27, elem28, elem29, elem30, elem31, elem32;
	T elem33, elem34, elem35, elem36;
	Sum op;

	/*cyclic distribution of thread blocks*/
	for (gbid = blockIdx.x; gbid < numBlocks; gbid += gridDim.x) {

		/*get the base address*/
		base = ELEMENTS_PER_THREAD * (gbid * blockDim.x + warpId * 32);

		/*load 8 elements per thread*/
		idx = base + laneId;
		elem = dataIn[idx];
		idx += 32;
		elem2 = dataIn[idx];
		idx += 32;
		elem3 = dataIn[idx];
		idx += 32;
		elem4 = dataIn[idx];
		idx += 32;
		elem5 = dataIn[idx];
		idx += 32;
		elem6 = dataIn[idx];
		idx += 32;
		elem7 = dataIn[idx];
		idx += 32;
		elem8 = dataIn[idx];
		idx += 32;
		elem9 = dataIn[idx];
		idx += 32;
		elem10 = dataIn[idx];
		idx += 32;
		elem11 = dataIn[idx];
		idx += 32;
		elem12 = dataIn[idx];
		idx += 32;
		elem13 = dataIn[idx];
		idx += 32;
		elem14 = dataIn[idx];
		idx += 32;
		elem15 = dataIn[idx];
		idx += 32;
		elem16 = dataIn[idx];
		idx += 32;
		elem17 = dataIn[idx];
		idx += 32;
		elem18 = dataIn[idx];
		idx += 32;
		elem19 = dataIn[idx];
		idx += 32;
		elem20 = dataIn[idx];
		idx += 32;
		elem21 = dataIn[idx];
		idx += 32;
		elem22 = dataIn[idx];
		idx += 32;
		elem23 = dataIn[idx];
		idx += 32;
		elem24 = dataIn[idx];
		idx += 32;
		elem25 = dataIn[idx];
		idx += 32;
		elem26 = dataIn[idx];
		idx += 32;
		elem27 = dataIn[idx];
		idx += 32;
		elem28 = dataIn[idx];
		idx += 32;
		elem29 = dataIn[idx];
		idx += 32;
		elem30 = dataIn[idx];
		idx += 32;
		elem31 = dataIn[idx];
		idx += 32;
		elem32 = dataIn[idx];
		idx += 32;
		elem33 = dataIn[idx];
		idx += 32;
		elem34 = dataIn[idx];
		idx += 32;
		elem35 = dataIn[idx];
		idx += 32;
		elem36 = dataIn[idx];

		/*perform inclusive scan for each row of the matrix of size 8 by 32*/
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the first row of the matrix*/
			val = __shfl_up(elem, i);
			if (laneId >= i) {
				elem = op(elem, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the second row of the matrix*/
			val = __shfl_up(elem2, i);
			if (laneId >= i) {
				elem2 = op(elem2, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the third row of the matrix*/
			val = __shfl_up(elem3, i);
			if (laneId >= i) {
				elem3 = op(elem3, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fourth row of the matrix*/
			val = __shfl_up(elem4, i);
			if (laneId >= i) {
				elem4 = op(elem4, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fifth row of the matrix*/
			val = __shfl_up(elem5, i);
			if (laneId >= i) {
				elem5 = op(elem5, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the sixth row of the matrix*/
			val = __shfl_up(elem6, i);
			if (laneId >= i) {
				elem6 = op(elem6, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the seventh row of the matrix*/
			val = __shfl_up(elem7, i);
			if (laneId >= i) {
				elem7 = op(elem7, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the eighth row of the matrix*/
			val = __shfl_up(elem8, i);
			if (laneId >= i) {
				elem8 = op(elem8, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem9, i);
			if (laneId >= i) {
				elem9 = op(elem9, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem10, i);
			if (laneId >= i) {
				elem10 = op(elem10, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem11, i);
			if (laneId >= i) {
				elem11 = op(elem11, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem12, i);
			if (laneId >= i) {
				elem12 = op(elem12, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem13, i);
			if (laneId >= i) {
				elem13 = op(elem13, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem14, i);
			if (laneId >= i) {
				elem14 = op(elem14, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem15, i);
			if (laneId >= i) {
				elem15 = op(elem15, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem16, i);
			if (laneId >= i) {
				elem16 = op(elem16, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem17, i);
			if (laneId >= i) {
				elem17 = op(elem17, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem18, i);
			if (laneId >= i) {
				elem18 = op(elem18, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem19, i);
			if (laneId >= i) {
				elem19 = op(elem19, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem20, i);
			if (laneId >= i) {
				elem20 = op(elem20, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem21, i);
			if (laneId >= i) {
				elem21 = op(elem21, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem22, i);
			if (laneId >= i) {
				elem22 = op(elem22, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem23, i);
			if (laneId >= i) {
				elem23 = op(elem23, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem24, i);
			if (laneId >= i) {
				elem24 = op(elem24, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem25, i);
			if (laneId >= i) {
				elem25 = op(elem25, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem26, i);
			if (laneId >= i) {
				elem26 = op(elem26, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem27, i);
			if (laneId >= i) {
				elem27 = op(elem27, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem28, i);
			if (laneId >= i) {
				elem28 = op(elem28, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem29, i);
			if (laneId >= i) {
				elem29 = op(elem29, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem30, i);
			if (laneId >= i) {
				elem30 = op(elem30, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem31, i);
			if (laneId >= i) {
				elem31 = op(elem31, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem32, i);
			if (laneId >= i) {
				elem32 = op(elem32, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem33, i);
			if (laneId >= i) {
				elem33 = op(elem33, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem34, i);
			if (laneId >= i) {
				elem34 = op(elem34, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem35, i);
			if (laneId >= i) {
				elem35 = op(elem35, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem36, i);
			if (laneId >= i) {
				elem36 = op(elem36, val);
			}
		}

		/*perform intra-warp inclusive scan by broadcasting the last column of the matrix to each individual thread*/
		elem2 = op(elem2, __shfl(elem, 31));
		elem3 = op(elem3, __shfl(elem2, 31));
		elem4 = op(elem4, __shfl(elem3, 31));
		elem5 = op(elem5, __shfl(elem4, 31));
		elem6 = op(elem6, __shfl(elem5, 31));
		elem7 = op(elem7, __shfl(elem6, 31));
		elem8 = op(elem8, __shfl(elem7, 31));
		elem9 = op(elem9, __shfl(elem8, 31));
		elem10 = op(elem10, __shfl(elem9, 31));
		elem11 = op(elem11, __shfl(elem10, 31));
		elem12 = op(elem12, __shfl(elem11, 31));
		elem13 = op(elem13, __shfl(elem12, 31));
		elem14 = op(elem14, __shfl(elem13, 31));
		elem15 = op(elem15, __shfl(elem14, 31));
		elem16 = op(elem16, __shfl(elem15, 31));
		elem17 = op(elem17, __shfl(elem16, 31));
		elem18 = op(elem18, __shfl(elem17, 31));
		elem19 = op(elem19, __shfl(elem18, 31));
		elem20 = op(elem20, __shfl(elem19, 31));
		elem21 = op(elem21, __shfl(elem20, 31));
		elem22 = op(elem22, __shfl(elem21, 31));
		elem23 = op(elem23, __shfl(elem22, 31));
		elem24 = op(elem24, __shfl(elem23, 31));
		elem25 = op(elem25, __shfl(elem24, 31));
		elem26 = op(elem26, __shfl(elem25, 31));
		elem27 = op(elem27, __shfl(elem26, 31));
		elem28 = op(elem28, __shfl(elem27, 31));
		elem29 = op(elem29, __shfl(elem28, 31));
		elem30 = op(elem30, __shfl(elem29, 31));
		elem31 = op(elem31, __shfl(elem30, 31));
		elem32 = op(elem32, __shfl(elem31, 31));
		elem33 = op(elem33, __shfl(elem32, 31));
		elem34 = op(elem34, __shfl(elem33, 31));
		elem35 = op(elem35, __shfl(elem34, 31));
		elem36 = op(elem36, __shfl(elem35, 31));

		/*save its sum to shared memory for each warp in the thread block*/
		if (laneId == 31) {
			shrdMem[warpId] = elem36;
		}
		__syncthreads();

		/*perform intra-block scan*/
		if (warpId == 0) {
			T warp = shrdMem[laneId];
#pragma unroll
			for (int i = 1; i <= 32; i *= 2) {
				val = __shfl_up(warp, i);
				if (laneId >= i) {
					warp += val;
				}
			}
			/*save the prefix sums to shared memory*/
			shrdMem[laneId] = warp;
		}
		__syncthreads();

		/*update each element in the thread block*/
		val = warpId == 0 ? 0 : shrdMem[warpId - 1];
		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);
		elem25 = op(elem25, val);
		elem26 = op(elem26, val);
		elem27 = op(elem27, val);
		elem28 = op(elem28, val);
		elem29 = op(elem29, val);
		elem30 = op(elem30, val);
		elem31 = op(elem31, val);
		elem32 = op(elem32, val);
		elem33 = op(elem33, val);
		elem34 = op(elem34, val);
		elem35 = op(elem35, val);
		elem36 = op(elem36, val);

		/*get the lead sum for the current thread block*/
		val = utils::busy_wait_comm<T, Comm>(partialSums, gbid, elem36);

		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);
		elem25 = op(elem25, val);
		elem26 = op(elem26, val);
		elem27 = op(elem27, val);
		elem28 = op(elem28, val);
		elem29 = op(elem29, val);
		elem30 = op(elem30, val);
		elem31 = op(elem31, val);
		elem32 = op(elem32, val);
		elem33 = op(elem33, val);
		elem34 = op(elem34, val);
		elem35 = op(elem35, val);
		elem36 = op(elem36, val);

		/*write the results to the output*/
		idx = base + laneId;
		dataOut[idx] = elem;
		idx += 32;
		dataOut[idx] = elem2;
		idx += 32;
		dataOut[idx] = elem3;
		idx += 32;
		dataOut[idx] = elem4;
		idx += 32;
		dataOut[idx] = elem5;
		idx += 32;
		dataOut[idx] = elem6;
		idx += 32;
		dataOut[idx] = elem7;
		idx += 32;
		dataOut[idx] = elem8;
		idx += 32;
		dataOut[idx] = elem9;
		idx += 32;
		dataOut[idx] = elem10;
		idx += 32;
		dataOut[idx] = elem11;
		idx += 32;
		dataOut[idx] = elem12;
		idx += 32;
		dataOut[idx] = elem13;
		idx += 32;
		dataOut[idx] = elem14;
		idx += 32;
		dataOut[idx] = elem15;
		idx += 32;
		dataOut[idx] = elem16;
		idx += 32;
		dataOut[idx] = elem17;
		idx += 32;
		dataOut[idx] = elem18;
		idx += 32;
		dataOut[idx] = elem19;
		idx += 32;
		dataOut[idx] = elem20;
		idx += 32;
		dataOut[idx] = elem21;
		idx += 32;
		dataOut[idx] = elem22;
		idx += 32;
		dataOut[idx] = elem23;
		idx += 32;
		dataOut[idx] = elem24;
		idx += 32;
		dataOut[idx] = elem25;
		idx += 32;
		dataOut[idx] = elem26;
		idx += 32;
		dataOut[idx] = elem27;
		idx += 32;
		dataOut[idx] = elem28;
		idx += 32;
		dataOut[idx] = elem29;
		idx += 32;
		dataOut[idx] = elem30;
		idx += 32;
		dataOut[idx] = elem31;
		idx += 32;
		dataOut[idx] = elem32;
		idx += 32;
		dataOut[idx] = elem33;
		idx += 32;
		dataOut[idx] = elem34;
		idx += 32;
		dataOut[idx] = elem35;
		idx += 32;
		dataOut[idx] = elem36;
	}
}

template<typename T, typename Sum, typename Comm, int ELEMENTS_PER_THREAD>
__device__ void priv_scan_stride_40(const int laneId, const int warpId, T* shrdMem,
		const T* __restrict dataIn, T* dataOut, Comm* partialSums,
		const unsigned int numBlocks) {
	int idx, gbid, base;
	T val, elem, elem2, elem3, elem4, elem5, elem6, elem7, elem8;
	T elem9, elem10, elem11, elem12, elem13, elem14, elem15, elem16;
	T elem17, elem18, elem19, elem20, elem21, elem22, elem23, elem24;
	T elem25, elem26, elem27, elem28, elem29, elem30, elem31, elem32;
	T elem33, elem34, elem35, elem36, elem37, elem38, elem39, elem40;
	Sum op;

	/*cyclic distribution of thread blocks*/
	for (gbid = blockIdx.x; gbid < numBlocks; gbid += gridDim.x) {

		/*get the base address*/
		base = ELEMENTS_PER_THREAD * (gbid * blockDim.x + warpId * 32);

		/*load 8 elements per thread*/
		idx = base + laneId;
		elem = dataIn[idx];
		idx += 32;
		elem2 = dataIn[idx];
		idx += 32;
		elem3 = dataIn[idx];
		idx += 32;
		elem4 = dataIn[idx];
		idx += 32;
		elem5 = dataIn[idx];
		idx += 32;
		elem6 = dataIn[idx];
		idx += 32;
		elem7 = dataIn[idx];
		idx += 32;
		elem8 = dataIn[idx];
		idx += 32;
		elem9 = dataIn[idx];
		idx += 32;
		elem10 = dataIn[idx];
		idx += 32;
		elem11 = dataIn[idx];
		idx += 32;
		elem12 = dataIn[idx];
		idx += 32;
		elem13 = dataIn[idx];
		idx += 32;
		elem14 = dataIn[idx];
		idx += 32;
		elem15 = dataIn[idx];
		idx += 32;
		elem16 = dataIn[idx];
		idx += 32;
		elem17 = dataIn[idx];
		idx += 32;
		elem18 = dataIn[idx];
		idx += 32;
		elem19 = dataIn[idx];
		idx += 32;
		elem20 = dataIn[idx];
		idx += 32;
		elem21 = dataIn[idx];
		idx += 32;
		elem22 = dataIn[idx];
		idx += 32;
		elem23 = dataIn[idx];
		idx += 32;
		elem24 = dataIn[idx];
		idx += 32;
		elem25 = dataIn[idx];
		idx += 32;
		elem26 = dataIn[idx];
		idx += 32;
		elem27 = dataIn[idx];
		idx += 32;
		elem28 = dataIn[idx];
		idx += 32;
		elem29 = dataIn[idx];
		idx += 32;
		elem30 = dataIn[idx];
		idx += 32;
		elem31 = dataIn[idx];
		idx += 32;
		elem32 = dataIn[idx];
		idx += 32;
		elem33 = dataIn[idx];
		idx += 32;
		elem34 = dataIn[idx];
		idx += 32;
		elem35 = dataIn[idx];
		idx += 32;
		elem36 = dataIn[idx];
		idx += 32;
		elem37 = dataIn[idx];
		idx += 32;
		elem38 = dataIn[idx];
		idx += 32;
		elem39 = dataIn[idx];
		idx += 32;
		elem40 = dataIn[idx];

		/*perform inclusive scan for each row of the matrix of size 8 by 32*/
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the first row of the matrix*/
			val = __shfl_up(elem, i);
			if (laneId >= i) {
				elem = op(elem, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the second row of the matrix*/
			val = __shfl_up(elem2, i);
			if (laneId >= i) {
				elem2 = op(elem2, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the third row of the matrix*/
			val = __shfl_up(elem3, i);
			if (laneId >= i) {
				elem3 = op(elem3, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fourth row of the matrix*/
			val = __shfl_up(elem4, i);
			if (laneId >= i) {
				elem4 = op(elem4, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fifth row of the matrix*/
			val = __shfl_up(elem5, i);
			if (laneId >= i) {
				elem5 = op(elem5, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the sixth row of the matrix*/
			val = __shfl_up(elem6, i);
			if (laneId >= i) {
				elem6 = op(elem6, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the seventh row of the matrix*/
			val = __shfl_up(elem7, i);
			if (laneId >= i) {
				elem7 = op(elem7, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the eighth row of the matrix*/
			val = __shfl_up(elem8, i);
			if (laneId >= i) {
				elem8 = op(elem8, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem9, i);
			if (laneId >= i) {
				elem9 = op(elem9, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem10, i);
			if (laneId >= i) {
				elem10 = op(elem10, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem11, i);
			if (laneId >= i) {
				elem11 = op(elem11, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem12, i);
			if (laneId >= i) {
				elem12 = op(elem12, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem13, i);
			if (laneId >= i) {
				elem13 = op(elem13, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem14, i);
			if (laneId >= i) {
				elem14 = op(elem14, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem15, i);
			if (laneId >= i) {
				elem15 = op(elem15, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem16, i);
			if (laneId >= i) {
				elem16 = op(elem16, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem17, i);
			if (laneId >= i) {
				elem17 = op(elem17, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem18, i);
			if (laneId >= i) {
				elem18 = op(elem18, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem19, i);
			if (laneId >= i) {
				elem19 = op(elem19, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem20, i);
			if (laneId >= i) {
				elem20 = op(elem20, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem21, i);
			if (laneId >= i) {
				elem21 = op(elem21, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem22, i);
			if (laneId >= i) {
				elem22 = op(elem22, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem23, i);
			if (laneId >= i) {
				elem23 = op(elem23, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem24, i);
			if (laneId >= i) {
				elem24 = op(elem24, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem25, i);
			if (laneId >= i) {
				elem25 = op(elem25, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem26, i);
			if (laneId >= i) {
				elem26 = op(elem26, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem27, i);
			if (laneId >= i) {
				elem27 = op(elem27, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem28, i);
			if (laneId >= i) {
				elem28 = op(elem28, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem29, i);
			if (laneId >= i) {
				elem29 = op(elem29, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem30, i);
			if (laneId >= i) {
				elem30 = op(elem30, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem31, i);
			if (laneId >= i) {
				elem31 = op(elem31, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem32, i);
			if (laneId >= i) {
				elem32 = op(elem32, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem33, i);
			if (laneId >= i) {
				elem33 = op(elem33, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem34, i);
			if (laneId >= i) {
				elem34 = op(elem34, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem35, i);
			if (laneId >= i) {
				elem35 = op(elem35, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem36, i);
			if (laneId >= i) {
				elem36 = op(elem36, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem37, i);
			if (laneId >= i) {
				elem37 = op(elem37, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem38, i);
			if (laneId >= i) {
				elem38 = op(elem38, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem39, i);
			if (laneId >= i) {
				elem39 = op(elem39, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem40, i);
			if (laneId >= i) {
				elem40 = op(elem40, val);
			}
		}

		/*perform intra-warp inclusive scan by broadcasting the last column of the matrix to each individual thread*/
		elem2 = op(elem2, __shfl(elem, 31));
		elem3 = op(elem3, __shfl(elem2, 31));
		elem4 = op(elem4, __shfl(elem3, 31));
		elem5 = op(elem5, __shfl(elem4, 31));
		elem6 = op(elem6, __shfl(elem5, 31));
		elem7 = op(elem7, __shfl(elem6, 31));
		elem8 = op(elem8, __shfl(elem7, 31));
		elem9 = op(elem9, __shfl(elem8, 31));
		elem10 = op(elem10, __shfl(elem9, 31));
		elem11 = op(elem11, __shfl(elem10, 31));
		elem12 = op(elem12, __shfl(elem11, 31));
		elem13 = op(elem13, __shfl(elem12, 31));
		elem14 = op(elem14, __shfl(elem13, 31));
		elem15 = op(elem15, __shfl(elem14, 31));
		elem16 = op(elem16, __shfl(elem15, 31));
		elem17 = op(elem17, __shfl(elem16, 31));
		elem18 = op(elem18, __shfl(elem17, 31));
		elem19 = op(elem19, __shfl(elem18, 31));
		elem20 = op(elem20, __shfl(elem19, 31));
		elem21 = op(elem21, __shfl(elem20, 31));
		elem22 = op(elem22, __shfl(elem21, 31));
		elem23 = op(elem23, __shfl(elem22, 31));
		elem24 = op(elem24, __shfl(elem23, 31));
		elem25 = op(elem25, __shfl(elem24, 31));
		elem26 = op(elem26, __shfl(elem25, 31));
		elem27 = op(elem27, __shfl(elem26, 31));
		elem28 = op(elem28, __shfl(elem27, 31));
		elem29 = op(elem29, __shfl(elem28, 31));
		elem30 = op(elem30, __shfl(elem29, 31));
		elem31 = op(elem31, __shfl(elem30, 31));
		elem32 = op(elem32, __shfl(elem31, 31));
		elem33 = op(elem33, __shfl(elem32, 31));
		elem34 = op(elem34, __shfl(elem33, 31));
		elem35 = op(elem35, __shfl(elem34, 31));
		elem36 = op(elem36, __shfl(elem35, 31));
		elem37 = op(elem37, __shfl(elem36, 31));
		elem38 = op(elem38, __shfl(elem37, 31));
		elem39 = op(elem39, __shfl(elem38, 31));
		elem40 = op(elem40, __shfl(elem39, 31));

		/*save its sum to shared memory for each warp in the thread block*/
		if (laneId == 31) {
			shrdMem[warpId] = elem40;
		}
		__syncthreads();

		/*perform intra-block scan*/
		if (warpId == 0) {
			T warp = shrdMem[laneId];
#pragma unroll
			for (int i = 1; i <= 32; i *= 2) {
				val = __shfl_up(warp, i);
				if (laneId >= i) {
					warp += val;
				}
			}
			/*save the prefix sums to shared memory*/
			shrdMem[laneId] = warp;
		}
		__syncthreads();

		/*update each element in the thread block*/
		val = warpId == 0 ? 0 : shrdMem[warpId - 1];
		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);
		elem25 = op(elem25, val);
		elem26 = op(elem26, val);
		elem27 = op(elem27, val);
		elem28 = op(elem28, val);
		elem29 = op(elem29, val);
		elem30 = op(elem30, val);
		elem31 = op(elem31, val);
		elem32 = op(elem32, val);
		elem33 = op(elem33, val);
		elem34 = op(elem34, val);
		elem35 = op(elem35, val);
		elem36 = op(elem36, val);
		elem37 = op(elem37, val);
		elem38 = op(elem38, val);
		elem39 = op(elem39, val);
		elem40 = op(elem40, val);

		/*get the lead sum for the current thread block*/
		val = utils::busy_wait_comm<T, Comm>(partialSums, gbid, elem40);

		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);
		elem25 = op(elem25, val);
		elem26 = op(elem26, val);
		elem27 = op(elem27, val);
		elem28 = op(elem28, val);
		elem29 = op(elem29, val);
		elem30 = op(elem30, val);
		elem31 = op(elem31, val);
		elem32 = op(elem32, val);
		elem33 = op(elem33, val);
		elem34 = op(elem34, val);
		elem35 = op(elem35, val);
		elem36 = op(elem36, val);
		elem37 = op(elem37, val);
		elem38 = op(elem38, val);
		elem39 = op(elem39, val);
		elem40 = op(elem40, val);

		/*write the results to the output*/
		idx = base + laneId;
		dataOut[idx] = elem;
		idx += 32;
		dataOut[idx] = elem2;
		idx += 32;
		dataOut[idx] = elem3;
		idx += 32;
		dataOut[idx] = elem4;
		idx += 32;
		dataOut[idx] = elem5;
		idx += 32;
		dataOut[idx] = elem6;
		idx += 32;
		dataOut[idx] = elem7;
		idx += 32;
		dataOut[idx] = elem8;
		idx += 32;
		dataOut[idx] = elem9;
		idx += 32;
		dataOut[idx] = elem10;
		idx += 32;
		dataOut[idx] = elem11;
		idx += 32;
		dataOut[idx] = elem12;
		idx += 32;
		dataOut[idx] = elem13;
		idx += 32;
		dataOut[idx] = elem14;
		idx += 32;
		dataOut[idx] = elem15;
		idx += 32;
		dataOut[idx] = elem16;
		idx += 32;
		dataOut[idx] = elem17;
		idx += 32;
		dataOut[idx] = elem18;
		idx += 32;
		dataOut[idx] = elem19;
		idx += 32;
		dataOut[idx] = elem20;
		idx += 32;
		dataOut[idx] = elem21;
		idx += 32;
		dataOut[idx] = elem22;
		idx += 32;
		dataOut[idx] = elem23;
		idx += 32;
		dataOut[idx] = elem24;
		idx += 32;
		dataOut[idx] = elem25;
		idx += 32;
		dataOut[idx] = elem26;
		idx += 32;
		dataOut[idx] = elem27;
		idx += 32;
		dataOut[idx] = elem28;
		idx += 32;
		dataOut[idx] = elem29;
		idx += 32;
		dataOut[idx] = elem30;
		idx += 32;
		dataOut[idx] = elem31;
		idx += 32;
		dataOut[idx] = elem32;
		idx += 32;
		dataOut[idx] = elem33;
		idx += 32;
		dataOut[idx] = elem34;
		idx += 32;
		dataOut[idx] = elem35;
		idx += 32;
		dataOut[idx] = elem36;
		idx += 32;
		dataOut[idx] = elem37;
		idx += 32;
		dataOut[idx] = elem38;
		idx += 32;
		dataOut[idx] = elem39;
		idx += 32;
		dataOut[idx] = elem40;
	}
}

template<typename T, typename Sum, typename Comm, int ELEMENTS_PER_THREAD>
__device__ void priv_scan_stride_44(const int laneId, const int warpId, T* shrdMem,
		const T* __restrict dataIn, T* dataOut, Comm* partialSums,
		const unsigned int numBlocks) {
	int idx, gbid, base;
	T val, elem, elem2, elem3, elem4, elem5, elem6, elem7, elem8;
	T elem9, elem10, elem11, elem12, elem13, elem14, elem15, elem16;
	T elem17, elem18, elem19, elem20, elem21, elem22, elem23, elem24;
	T elem25, elem26, elem27, elem28, elem29, elem30, elem31, elem32;
	T elem33, elem34, elem35, elem36, elem37, elem38, elem39, elem40;
	T elem41, elem42, elem43, elem44;
	Sum op;

	/*cyclic distribution of thread blocks*/
	for (gbid = blockIdx.x; gbid < numBlocks; gbid += gridDim.x) {

		/*get the base address*/
		base = ELEMENTS_PER_THREAD * (gbid * blockDim.x + warpId * 32);

		/*load 8 elements per thread*/
		idx = base + laneId;
		elem = dataIn[idx];
		idx += 32;
		elem2 = dataIn[idx];
		idx += 32;
		elem3 = dataIn[idx];
		idx += 32;
		elem4 = dataIn[idx];
		idx += 32;
		elem5 = dataIn[idx];
		idx += 32;
		elem6 = dataIn[idx];
		idx += 32;
		elem7 = dataIn[idx];
		idx += 32;
		elem8 = dataIn[idx];
		idx += 32;
		elem9 = dataIn[idx];
		idx += 32;
		elem10 = dataIn[idx];
		idx += 32;
		elem11 = dataIn[idx];
		idx += 32;
		elem12 = dataIn[idx];
		idx += 32;
		elem13 = dataIn[idx];
		idx += 32;
		elem14 = dataIn[idx];
		idx += 32;
		elem15 = dataIn[idx];
		idx += 32;
		elem16 = dataIn[idx];
		idx += 32;
		elem17 = dataIn[idx];
		idx += 32;
		elem18 = dataIn[idx];
		idx += 32;
		elem19 = dataIn[idx];
		idx += 32;
		elem20 = dataIn[idx];
		idx += 32;
		elem21 = dataIn[idx];
		idx += 32;
		elem22 = dataIn[idx];
		idx += 32;
		elem23 = dataIn[idx];
		idx += 32;
		elem24 = dataIn[idx];
		idx += 32;
		elem25 = dataIn[idx];
		idx += 32;
		elem26 = dataIn[idx];
		idx += 32;
		elem27 = dataIn[idx];
		idx += 32;
		elem28 = dataIn[idx];
		idx += 32;
		elem29 = dataIn[idx];
		idx += 32;
		elem30 = dataIn[idx];
		idx += 32;
		elem31 = dataIn[idx];
		idx += 32;
		elem32 = dataIn[idx];
		idx += 32;
		elem33 = dataIn[idx];
		idx += 32;
		elem34 = dataIn[idx];
		idx += 32;
		elem35 = dataIn[idx];
		idx += 32;
		elem36 = dataIn[idx];
		idx += 32;
		elem37 = dataIn[idx];
		idx += 32;
		elem38 = dataIn[idx];
		idx += 32;
		elem39 = dataIn[idx];
		idx += 32;
		elem40 = dataIn[idx];
		idx += 32;
		elem41 = dataIn[idx];
		idx += 32;
		elem42 = dataIn[idx];
		idx += 32;
		elem43 = dataIn[idx];
		idx += 32;
		elem44 = dataIn[idx];

		/*perform inclusive scan for each row of the matrix of size 8 by 32*/
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the first row of the matrix*/
			val = __shfl_up(elem, i);
			if (laneId >= i) {
				elem = op(elem, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the second row of the matrix*/
			val = __shfl_up(elem2, i);
			if (laneId >= i) {
				elem2 = op(elem2, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the third row of the matrix*/
			val = __shfl_up(elem3, i);
			if (laneId >= i) {
				elem3 = op(elem3, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fourth row of the matrix*/
			val = __shfl_up(elem4, i);
			if (laneId >= i) {
				elem4 = op(elem4, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fifth row of the matrix*/
			val = __shfl_up(elem5, i);
			if (laneId >= i) {
				elem5 = op(elem5, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the sixth row of the matrix*/
			val = __shfl_up(elem6, i);
			if (laneId >= i) {
				elem6 = op(elem6, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the seventh row of the matrix*/
			val = __shfl_up(elem7, i);
			if (laneId >= i) {
				elem7 = op(elem7, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the eighth row of the matrix*/
			val = __shfl_up(elem8, i);
			if (laneId >= i) {
				elem8 = op(elem8, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem9, i);
			if (laneId >= i) {
				elem9 = op(elem9, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem10, i);
			if (laneId >= i) {
				elem10 = op(elem10, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem11, i);
			if (laneId >= i) {
				elem11 = op(elem11, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem12, i);
			if (laneId >= i) {
				elem12 = op(elem12, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem13, i);
			if (laneId >= i) {
				elem13 = op(elem13, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem14, i);
			if (laneId >= i) {
				elem14 = op(elem14, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem15, i);
			if (laneId >= i) {
				elem15 = op(elem15, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem16, i);
			if (laneId >= i) {
				elem16 = op(elem16, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem17, i);
			if (laneId >= i) {
				elem17 = op(elem17, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem18, i);
			if (laneId >= i) {
				elem18 = op(elem18, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem19, i);
			if (laneId >= i) {
				elem19 = op(elem19, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem20, i);
			if (laneId >= i) {
				elem20 = op(elem20, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem21, i);
			if (laneId >= i) {
				elem21 = op(elem21, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem22, i);
			if (laneId >= i) {
				elem22 = op(elem22, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem23, i);
			if (laneId >= i) {
				elem23 = op(elem23, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem24, i);
			if (laneId >= i) {
				elem24 = op(elem24, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem25, i);
			if (laneId >= i) {
				elem25 = op(elem25, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem26, i);
			if (laneId >= i) {
				elem26 = op(elem26, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem27, i);
			if (laneId >= i) {
				elem27 = op(elem27, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem28, i);
			if (laneId >= i) {
				elem28 = op(elem28, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem29, i);
			if (laneId >= i) {
				elem29 = op(elem29, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem30, i);
			if (laneId >= i) {
				elem30 = op(elem30, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem31, i);
			if (laneId >= i) {
				elem31 = op(elem31, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem32, i);
			if (laneId >= i) {
				elem32 = op(elem32, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem33, i);
			if (laneId >= i) {
				elem33 = op(elem33, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem34, i);
			if (laneId >= i) {
				elem34 = op(elem34, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem35, i);
			if (laneId >= i) {
				elem35 = op(elem35, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem36, i);
			if (laneId >= i) {
				elem36 = op(elem36, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem37, i);
			if (laneId >= i) {
				elem37 = op(elem37, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem38, i);
			if (laneId >= i) {
				elem38 = op(elem38, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem39, i);
			if (laneId >= i) {
				elem39 = op(elem39, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem40, i);
			if (laneId >= i) {
				elem40 = op(elem40, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem41, i);
			if (laneId >= i) {
				elem41 = op(elem41, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem42, i);
			if (laneId >= i) {
				elem42 = op(elem42, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem43, i);
			if (laneId >= i) {
				elem43 = op(elem43, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem44, i);
			if (laneId >= i) {
				elem44 = op(elem44, val);
			}
		}

		/*perform intra-warp inclusive scan by broadcasting the last column of the matrix to each individual thread*/
		elem2 = op(elem2, __shfl(elem, 31));
		elem3 = op(elem3, __shfl(elem2, 31));
		elem4 = op(elem4, __shfl(elem3, 31));
		elem5 = op(elem5, __shfl(elem4, 31));
		elem6 = op(elem6, __shfl(elem5, 31));
		elem7 = op(elem7, __shfl(elem6, 31));
		elem8 = op(elem8, __shfl(elem7, 31));
		elem9 = op(elem9, __shfl(elem8, 31));
		elem10 = op(elem10, __shfl(elem9, 31));
		elem11 = op(elem11, __shfl(elem10, 31));
		elem12 = op(elem12, __shfl(elem11, 31));
		elem13 = op(elem13, __shfl(elem12, 31));
		elem14 = op(elem14, __shfl(elem13, 31));
		elem15 = op(elem15, __shfl(elem14, 31));
		elem16 = op(elem16, __shfl(elem15, 31));
		elem17 = op(elem17, __shfl(elem16, 31));
		elem18 = op(elem18, __shfl(elem17, 31));
		elem19 = op(elem19, __shfl(elem18, 31));
		elem20 = op(elem20, __shfl(elem19, 31));
		elem21 = op(elem21, __shfl(elem20, 31));
		elem22 = op(elem22, __shfl(elem21, 31));
		elem23 = op(elem23, __shfl(elem22, 31));
		elem24 = op(elem24, __shfl(elem23, 31));
		elem25 = op(elem25, __shfl(elem24, 31));
		elem26 = op(elem26, __shfl(elem25, 31));
		elem27 = op(elem27, __shfl(elem26, 31));
		elem28 = op(elem28, __shfl(elem27, 31));
		elem29 = op(elem29, __shfl(elem28, 31));
		elem30 = op(elem30, __shfl(elem29, 31));
		elem31 = op(elem31, __shfl(elem30, 31));
		elem32 = op(elem32, __shfl(elem31, 31));
		elem33 = op(elem33, __shfl(elem32, 31));
		elem34 = op(elem34, __shfl(elem33, 31));
		elem35 = op(elem35, __shfl(elem34, 31));
		elem36 = op(elem36, __shfl(elem35, 31));
		elem37 = op(elem37, __shfl(elem36, 31));
		elem38 = op(elem38, __shfl(elem37, 31));
		elem39 = op(elem39, __shfl(elem38, 31));
		elem40 = op(elem40, __shfl(elem39, 31));
		elem41 = op(elem41, __shfl(elem40, 31));
		elem42 = op(elem42, __shfl(elem41, 31));
		elem43 = op(elem43, __shfl(elem42, 31));
		elem44 = op(elem44, __shfl(elem43, 31));

		/*save its sum to shared memory for each warp in the thread block*/
		if (laneId == 31) {
			shrdMem[warpId] = elem44;
		}
		__syncthreads();

		/*perform intra-block scan*/
		if (warpId == 0) {
			T warp = shrdMem[laneId];
#pragma unroll
			for (int i = 1; i <= 32; i *= 2) {
				val = __shfl_up(warp, i);
				if (laneId >= i) {
					warp += val;
				}
			}
			/*save the prefix sums to shared memory*/
			shrdMem[laneId] = warp;
		}
		__syncthreads();

		/*update each element in the thread block*/
		val = warpId == 0 ? 0 : shrdMem[warpId - 1];
		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);
		elem25 = op(elem25, val);
		elem26 = op(elem26, val);
		elem27 = op(elem27, val);
		elem28 = op(elem28, val);
		elem29 = op(elem29, val);
		elem30 = op(elem30, val);
		elem31 = op(elem31, val);
		elem32 = op(elem32, val);
		elem33 = op(elem33, val);
		elem34 = op(elem34, val);
		elem35 = op(elem35, val);
		elem36 = op(elem36, val);
		elem37 = op(elem37, val);
		elem38 = op(elem38, val);
		elem39 = op(elem39, val);
		elem40 = op(elem40, val);
		elem41 = op(elem41, val);
		elem42 = op(elem42, val);
		elem43 = op(elem43, val);
		elem44 = op(elem44, val);

		/*get the lead sum for the current thread block*/
		val = utils::busy_wait_comm<T, Comm>(partialSums, gbid, elem44);

		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);
		elem25 = op(elem25, val);
		elem26 = op(elem26, val);
		elem27 = op(elem27, val);
		elem28 = op(elem28, val);
		elem29 = op(elem29, val);
		elem30 = op(elem30, val);
		elem31 = op(elem31, val);
		elem32 = op(elem32, val);
		elem33 = op(elem33, val);
		elem34 = op(elem34, val);
		elem35 = op(elem35, val);
		elem36 = op(elem36, val);
		elem37 = op(elem37, val);
		elem38 = op(elem38, val);
		elem39 = op(elem39, val);
		elem40 = op(elem40, val);
		elem41 = op(elem41, val);
		elem42 = op(elem42, val);
		elem43 = op(elem43, val);
		elem44 = op(elem44, val);

		/*write the results to the output*/
		idx = base + laneId;
		dataOut[idx] = elem;
		idx += 32;
		dataOut[idx] = elem2;
		idx += 32;
		dataOut[idx] = elem3;
		idx += 32;
		dataOut[idx] = elem4;
		idx += 32;
		dataOut[idx] = elem5;
		idx += 32;
		dataOut[idx] = elem6;
		idx += 32;
		dataOut[idx] = elem7;
		idx += 32;
		dataOut[idx] = elem8;
		idx += 32;
		dataOut[idx] = elem9;
		idx += 32;
		dataOut[idx] = elem10;
		idx += 32;
		dataOut[idx] = elem11;
		idx += 32;
		dataOut[idx] = elem12;
		idx += 32;
		dataOut[idx] = elem13;
		idx += 32;
		dataOut[idx] = elem14;
		idx += 32;
		dataOut[idx] = elem15;
		idx += 32;
		dataOut[idx] = elem16;
		idx += 32;
		dataOut[idx] = elem17;
		idx += 32;
		dataOut[idx] = elem18;
		idx += 32;
		dataOut[idx] = elem19;
		idx += 32;
		dataOut[idx] = elem20;
		idx += 32;
		dataOut[idx] = elem21;
		idx += 32;
		dataOut[idx] = elem22;
		idx += 32;
		dataOut[idx] = elem23;
		idx += 32;
		dataOut[idx] = elem24;
		idx += 32;
		dataOut[idx] = elem25;
		idx += 32;
		dataOut[idx] = elem26;
		idx += 32;
		dataOut[idx] = elem27;
		idx += 32;
		dataOut[idx] = elem28;
		idx += 32;
		dataOut[idx] = elem29;
		idx += 32;
		dataOut[idx] = elem30;
		idx += 32;
		dataOut[idx] = elem31;
		idx += 32;
		dataOut[idx] = elem32;
		idx += 32;
		dataOut[idx] = elem33;
		idx += 32;
		dataOut[idx] = elem34;
		idx += 32;
		dataOut[idx] = elem35;
		idx += 32;
		dataOut[idx] = elem36;
		idx += 32;
		dataOut[idx] = elem37;
		idx += 32;
		dataOut[idx] = elem38;
		idx += 32;
		dataOut[idx] = elem39;
		idx += 32;
		dataOut[idx] = elem40;
		idx += 32;
		dataOut[idx] = elem41;
		idx += 32;
		dataOut[idx] = elem42;
		idx += 32;
		dataOut[idx] = elem43;
		idx += 32;
		dataOut[idx] = elem44;
	}
}

template<typename T, typename Sum, typename Comm, int ELEMENTS_PER_THREAD>
__device__ void priv_scan_stride_48(const int laneId, const int warpId, T* shrdMem,
		const T* __restrict dataIn, T* dataOut, Comm* partialSums,
		const unsigned int numBlocks) {
	int idx, gbid, base;
	T val, elem, elem2, elem3, elem4, elem5, elem6, elem7, elem8;
	T elem9, elem10, elem11, elem12, elem13, elem14, elem15, elem16;
	T elem17, elem18, elem19, elem20, elem21, elem22, elem23, elem24;
	T elem25, elem26, elem27, elem28, elem29, elem30, elem31, elem32;
	T elem33, elem34, elem35, elem36, elem37, elem38, elem39, elem40;
	T elem41, elem42, elem43, elem44, elem45, elem46, elem47, elem48;
	Sum op;

	/*cyclic distribution of thread blocks*/
	for (gbid = blockIdx.x; gbid < numBlocks; gbid += gridDim.x) {

		/*get the base address*/
		base = ELEMENTS_PER_THREAD * (gbid * blockDim.x + warpId * 32);

		/*load 8 elements per thread*/
		idx = base + laneId;
		elem = dataIn[idx];
		idx += 32;
		elem2 = dataIn[idx];
		idx += 32;
		elem3 = dataIn[idx];
		idx += 32;
		elem4 = dataIn[idx];
		idx += 32;
		elem5 = dataIn[idx];
		idx += 32;
		elem6 = dataIn[idx];
		idx += 32;
		elem7 = dataIn[idx];
		idx += 32;
		elem8 = dataIn[idx];
		idx += 32;
		elem9 = dataIn[idx];
		idx += 32;
		elem10 = dataIn[idx];
		idx += 32;
		elem11 = dataIn[idx];
		idx += 32;
		elem12 = dataIn[idx];
		idx += 32;
		elem13 = dataIn[idx];
		idx += 32;
		elem14 = dataIn[idx];
		idx += 32;
		elem15 = dataIn[idx];
		idx += 32;
		elem16 = dataIn[idx];
		idx += 32;
		elem17 = dataIn[idx];
		idx += 32;
		elem18 = dataIn[idx];
		idx += 32;
		elem19 = dataIn[idx];
		idx += 32;
		elem20 = dataIn[idx];
		idx += 32;
		elem21 = dataIn[idx];
		idx += 32;
		elem22 = dataIn[idx];
		idx += 32;
		elem23 = dataIn[idx];
		idx += 32;
		elem24 = dataIn[idx];
		idx += 32;
		elem25 = dataIn[idx];
		idx += 32;
		elem26 = dataIn[idx];
		idx += 32;
		elem27 = dataIn[idx];
		idx += 32;
		elem28 = dataIn[idx];
		idx += 32;
		elem29 = dataIn[idx];
		idx += 32;
		elem30 = dataIn[idx];
		idx += 32;
		elem31 = dataIn[idx];
		idx += 32;
		elem32 = dataIn[idx];
		idx += 32;
		elem33 = dataIn[idx];
		idx += 32;
		elem34 = dataIn[idx];
		idx += 32;
		elem35 = dataIn[idx];
		idx += 32;
		elem36 = dataIn[idx];
		idx += 32;
		elem37 = dataIn[idx];
		idx += 32;
		elem38 = dataIn[idx];
		idx += 32;
		elem39 = dataIn[idx];
		idx += 32;
		elem40 = dataIn[idx];
		idx += 32;
		elem41 = dataIn[idx];
		idx += 32;
		elem42 = dataIn[idx];
		idx += 32;
		elem43 = dataIn[idx];
		idx += 32;
		elem44 = dataIn[idx];
		idx += 32;
		elem45 = dataIn[idx];
		idx += 32;
		elem46 = dataIn[idx];
		idx += 32;
		elem47 = dataIn[idx];
		idx += 32;
		elem48 = dataIn[idx];

		/*perform inclusive scan for each row of the matrix of size 8 by 32*/
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the first row of the matrix*/
			val = __shfl_up(elem, i);
			if (laneId >= i) {
				elem = op(elem, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the second row of the matrix*/
			val = __shfl_up(elem2, i);
			if (laneId >= i) {
				elem2 = op(elem2, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the third row of the matrix*/
			val = __shfl_up(elem3, i);
			if (laneId >= i) {
				elem3 = op(elem3, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fourth row of the matrix*/
			val = __shfl_up(elem4, i);
			if (laneId >= i) {
				elem4 = op(elem4, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the fifth row of the matrix*/
			val = __shfl_up(elem5, i);
			if (laneId >= i) {
				elem5 = op(elem5, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the sixth row of the matrix*/
			val = __shfl_up(elem6, i);
			if (laneId >= i) {
				elem6 = op(elem6, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the seventh row of the matrix*/
			val = __shfl_up(elem7, i);
			if (laneId >= i) {
				elem7 = op(elem7, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			/*the eighth row of the matrix*/
			val = __shfl_up(elem8, i);
			if (laneId >= i) {
				elem8 = op(elem8, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem9, i);
			if (laneId >= i) {
				elem9 = op(elem9, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem10, i);
			if (laneId >= i) {
				elem10 = op(elem10, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem11, i);
			if (laneId >= i) {
				elem11 = op(elem11, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem12, i);
			if (laneId >= i) {
				elem12 = op(elem12, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem13, i);
			if (laneId >= i) {
				elem13 = op(elem13, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem14, i);
			if (laneId >= i) {
				elem14 = op(elem14, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem15, i);
			if (laneId >= i) {
				elem15 = op(elem15, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem16, i);
			if (laneId >= i) {
				elem16 = op(elem16, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem17, i);
			if (laneId >= i) {
				elem17 = op(elem17, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem18, i);
			if (laneId >= i) {
				elem18 = op(elem18, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem19, i);
			if (laneId >= i) {
				elem19 = op(elem19, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem20, i);
			if (laneId >= i) {
				elem20 = op(elem20, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem21, i);
			if (laneId >= i) {
				elem21 = op(elem21, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem22, i);
			if (laneId >= i) {
				elem22 = op(elem22, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem23, i);
			if (laneId >= i) {
				elem23 = op(elem23, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem24, i);
			if (laneId >= i) {
				elem24 = op(elem24, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem25, i);
			if (laneId >= i) {
				elem25 = op(elem25, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem26, i);
			if (laneId >= i) {
				elem26 = op(elem26, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem27, i);
			if (laneId >= i) {
				elem27 = op(elem27, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem28, i);
			if (laneId >= i) {
				elem28 = op(elem28, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem29, i);
			if (laneId >= i) {
				elem29 = op(elem29, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem30, i);
			if (laneId >= i) {
				elem30 = op(elem30, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem31, i);
			if (laneId >= i) {
				elem31 = op(elem31, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem32, i);
			if (laneId >= i) {
				elem32 = op(elem32, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem33, i);
			if (laneId >= i) {
				elem33 = op(elem33, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem34, i);
			if (laneId >= i) {
				elem34 = op(elem34, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem35, i);
			if (laneId >= i) {
				elem35 = op(elem35, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem36, i);
			if (laneId >= i) {
				elem36 = op(elem36, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem37, i);
			if (laneId >= i) {
				elem37 = op(elem37, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem38, i);
			if (laneId >= i) {
				elem38 = op(elem38, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem39, i);
			if (laneId >= i) {
				elem39 = op(elem39, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem40, i);
			if (laneId >= i) {
				elem40 = op(elem40, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem41, i);
			if (laneId >= i) {
				elem41 = op(elem41, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem42, i);
			if (laneId >= i) {
				elem42 = op(elem42, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem43, i);
			if (laneId >= i) {
				elem43 = op(elem43, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem44, i);
			if (laneId >= i) {
				elem44 = op(elem44, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem45, i);
			if (laneId >= i) {
				elem45 = op(elem45, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem46, i);
			if (laneId >= i) {
				elem46 = op(elem46, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem47, i);
			if (laneId >= i) {
				elem47 = op(elem47, val);
			}
		}
#pragma unroll
		for (int i = 1; i <= 32; i *= 2) {
			val = __shfl_up(elem48, i);
			if (laneId >= i) {
				elem48 = op(elem48, val);
			}
		}

		/*perform intra-warp inclusive scan by broadcasting the last column of the matrix to each individual thread*/
		elem2 = op(elem2, __shfl(elem, 31));
		elem3 = op(elem3, __shfl(elem2, 31));
		elem4 = op(elem4, __shfl(elem3, 31));
		elem5 = op(elem5, __shfl(elem4, 31));
		elem6 = op(elem6, __shfl(elem5, 31));
		elem7 = op(elem7, __shfl(elem6, 31));
		elem8 = op(elem8, __shfl(elem7, 31));
		elem9 = op(elem9, __shfl(elem8, 31));
		elem10 = op(elem10, __shfl(elem9, 31));
		elem11 = op(elem11, __shfl(elem10, 31));
		elem12 = op(elem12, __shfl(elem11, 31));
		elem13 = op(elem13, __shfl(elem12, 31));
		elem14 = op(elem14, __shfl(elem13, 31));
		elem15 = op(elem15, __shfl(elem14, 31));
		elem16 = op(elem16, __shfl(elem15, 31));
		elem17 = op(elem17, __shfl(elem16, 31));
		elem18 = op(elem18, __shfl(elem17, 31));
		elem19 = op(elem19, __shfl(elem18, 31));
		elem20 = op(elem20, __shfl(elem19, 31));
		elem21 = op(elem21, __shfl(elem20, 31));
		elem22 = op(elem22, __shfl(elem21, 31));
		elem23 = op(elem23, __shfl(elem22, 31));
		elem24 = op(elem24, __shfl(elem23, 31));
		elem25 = op(elem25, __shfl(elem24, 31));
		elem26 = op(elem26, __shfl(elem25, 31));
		elem27 = op(elem27, __shfl(elem26, 31));
		elem28 = op(elem28, __shfl(elem27, 31));
		elem29 = op(elem29, __shfl(elem28, 31));
		elem30 = op(elem30, __shfl(elem29, 31));
		elem31 = op(elem31, __shfl(elem30, 31));
		elem32 = op(elem32, __shfl(elem31, 31));
		elem33 = op(elem33, __shfl(elem32, 31));
		elem34 = op(elem34, __shfl(elem33, 31));
		elem35 = op(elem35, __shfl(elem34, 31));
		elem36 = op(elem36, __shfl(elem35, 31));
		elem37 = op(elem37, __shfl(elem36, 31));
		elem38 = op(elem38, __shfl(elem37, 31));
		elem39 = op(elem39, __shfl(elem38, 31));
		elem40 = op(elem40, __shfl(elem39, 31));
		elem41 = op(elem41, __shfl(elem40, 31));
		elem42 = op(elem42, __shfl(elem41, 31));
		elem43 = op(elem43, __shfl(elem42, 31));
		elem44 = op(elem44, __shfl(elem43, 31));
		elem45 = op(elem45, __shfl(elem44, 31));
		elem46 = op(elem46, __shfl(elem45, 31));
		elem47 = op(elem47, __shfl(elem46, 31));
		elem48 = op(elem48, __shfl(elem47, 31));

		/*save its sum to shared memory for each warp in the thread block*/
		if (laneId == 31) {
			shrdMem[warpId] = elem48;
		}
		__syncthreads();

		/*perform intra-block scan*/
		if (warpId == 0) {
			T warp = shrdMem[laneId];
#pragma unroll
			for (int i = 1; i <= 32; i *= 2) {
				val = __shfl_up(warp, i);
				if (laneId >= i) {
					warp += val;
				}
			}
			/*save the prefix sums to shared memory*/
			shrdMem[laneId] = warp;
		}
		__syncthreads();

		/*update each element in the thread block*/
		val = warpId == 0 ? 0 : shrdMem[warpId - 1];
		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);
		elem25 = op(elem25, val);
		elem26 = op(elem26, val);
		elem27 = op(elem27, val);
		elem28 = op(elem28, val);
		elem29 = op(elem29, val);
		elem30 = op(elem30, val);
		elem31 = op(elem31, val);
		elem32 = op(elem32, val);
		elem33 = op(elem33, val);
		elem34 = op(elem34, val);
		elem35 = op(elem35, val);
		elem36 = op(elem36, val);
		elem37 = op(elem37, val);
		elem38 = op(elem38, val);
		elem39 = op(elem39, val);
		elem40 = op(elem40, val);
		elem41 = op(elem41, val);
		elem42 = op(elem42, val);
		elem43 = op(elem43, val);
		elem44 = op(elem44, val);
		elem45 = op(elem45, val);
		elem46 = op(elem46, val);
		elem47 = op(elem47, val);
		elem48 = op(elem48, val);

		/*get the lead sum for the current thread block*/
		val = utils::busy_wait_comm<T, Comm>(partialSums, gbid, elem48);

		elem = op(elem, val);
		elem2 = op(elem2, val);
		elem3 = op(elem3, val);
		elem4 = op(elem4, val);
		elem5 = op(elem5, val);
		elem6 = op(elem6, val);
		elem7 = op(elem7, val);
		elem8 = op(elem8, val);
		elem9 = op(elem9, val);
		elem10 = op(elem10, val);
		elem11 = op(elem11, val);
		elem12 = op(elem12, val);
		elem13 = op(elem13, val);
		elem14 = op(elem14, val);
		elem15 = op(elem15, val);
		elem16 = op(elem16, val);
		elem17 = op(elem17, val);
		elem18 = op(elem18, val);
		elem19 = op(elem19, val);
		elem20 = op(elem20, val);
		elem21 = op(elem21, val);
		elem22 = op(elem22, val);
		elem23 = op(elem23, val);
		elem24 = op(elem24, val);
		elem25 = op(elem25, val);
		elem26 = op(elem26, val);
		elem27 = op(elem27, val);
		elem28 = op(elem28, val);
		elem29 = op(elem29, val);
		elem30 = op(elem30, val);
		elem31 = op(elem31, val);
		elem32 = op(elem32, val);
		elem33 = op(elem33, val);
		elem34 = op(elem34, val);
		elem35 = op(elem35, val);
		elem36 = op(elem36, val);
		elem37 = op(elem37, val);
		elem38 = op(elem38, val);
		elem39 = op(elem39, val);
		elem40 = op(elem40, val);
		elem41 = op(elem41, val);
		elem42 = op(elem42, val);
		elem43 = op(elem43, val);
		elem44 = op(elem44, val);
		elem45 = op(elem45, val);
		elem46 = op(elem46, val);
		elem47 = op(elem47, val);
		elem48 = op(elem48, val);

		/*write the results to the output*/
		idx = base + laneId;
		dataOut[idx] = elem;
		idx += 32;
		dataOut[idx] = elem2;
		idx += 32;
		dataOut[idx] = elem3;
		idx += 32;
		dataOut[idx] = elem4;
		idx += 32;
		dataOut[idx] = elem5;
		idx += 32;
		dataOut[idx] = elem6;
		idx += 32;
		dataOut[idx] = elem7;
		idx += 32;
		dataOut[idx] = elem8;
		idx += 32;
		dataOut[idx] = elem9;
		idx += 32;
		dataOut[idx] = elem10;
		idx += 32;
		dataOut[idx] = elem11;
		idx += 32;
		dataOut[idx] = elem12;
		idx += 32;
		dataOut[idx] = elem13;
		idx += 32;
		dataOut[idx] = elem14;
		idx += 32;
		dataOut[idx] = elem15;
		idx += 32;
		dataOut[idx] = elem16;
		idx += 32;
		dataOut[idx] = elem17;
		idx += 32;
		dataOut[idx] = elem18;
		idx += 32;
		dataOut[idx] = elem19;
		idx += 32;
		dataOut[idx] = elem20;
		idx += 32;
		dataOut[idx] = elem21;
		idx += 32;
		dataOut[idx] = elem22;
		idx += 32;
		dataOut[idx] = elem23;
		idx += 32;
		dataOut[idx] = elem24;
		idx += 32;
		dataOut[idx] = elem25;
		idx += 32;
		dataOut[idx] = elem26;
		idx += 32;
		dataOut[idx] = elem27;
		idx += 32;
		dataOut[idx] = elem28;
		idx += 32;
		dataOut[idx] = elem29;
		idx += 32;
		dataOut[idx] = elem30;
		idx += 32;
		dataOut[idx] = elem31;
		idx += 32;
		dataOut[idx] = elem32;
		idx += 32;
		dataOut[idx] = elem33;
		idx += 32;
		dataOut[idx] = elem34;
		idx += 32;
		dataOut[idx] = elem35;
		idx += 32;
		dataOut[idx] = elem36;
		idx += 32;
		dataOut[idx] = elem37;
		idx += 32;
		dataOut[idx] = elem38;
		idx += 32;
		dataOut[idx] = elem39;
		idx += 32;
		dataOut[idx] = elem40;
		idx += 32;
		dataOut[idx] = elem41;
		idx += 32;
		dataOut[idx] = elem42;
		idx += 32;
		dataOut[idx] = elem43;
		idx += 32;
		dataOut[idx] = elem44;
		idx += 32;
		dataOut[idx] = elem45;
		idx += 32;
		dataOut[idx] = elem46;
		idx += 32;
		dataOut[idx] = elem47;
		idx += 32;
		dataOut[idx] = elem48;
	}
}

template<typename T, typename Sum, typename Comm>
__global__ void priv_scan_kernel_4(const T* __restrict dataIn, T* dataOut,
		const int numBlocks, Comm* partialSums) {
	__shared__ T shrdMem[32];
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	priv_scan_stride_4<T, Sum, Comm, 4>(laneId, warpId, shrdMem, dataIn, dataOut,
			partialSums, numBlocks);
}
template<typename T, typename Sum, typename Comm>
__global__ void priv_scan_kernel_8(const T* __restrict dataIn, T* dataOut,
		const int numBlocks, Comm* partialSums) {
	__shared__ T shrdMem[32];
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;

	priv_scan_stride_8<T, Sum, Comm, 8>(laneId, warpId, shrdMem, dataIn, dataOut,
			partialSums, numBlocks);
}
template<typename T, typename Sum, typename Comm>
__global__ void priv_scan_kernel_16(const T* __restrict dataIn, T* dataOut,
		const int numBlocks, Comm* partialSums) {
	__shared__ T shrdMem[32];
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;

	priv_scan_stride_16<T, Sum, Comm, 16>(laneId, warpId, shrdMem, dataIn, dataOut,
			partialSums, numBlocks);
}

template<typename T, typename Sum, typename Comm>
__global__ void priv_scan_kernel_20(const T* __restrict dataIn, T* dataOut,
		const int numBlocks, Comm* partialSums) {
	__shared__ T shrdMem[32];
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;

	priv_scan_stride_20<T, Sum, Comm, 20>(laneId, warpId, shrdMem, dataIn, dataOut,
			partialSums, numBlocks);
}
template<typename T, typename Sum, typename Comm>
__global__ void priv_scan_kernel_24(const T* __restrict dataIn, T* dataOut,
		const int numBlocks, Comm* partialSums) {
	__shared__ T shrdMem[32];
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;

	priv_scan_stride_24<T, Sum, Comm, 24>(laneId, warpId, shrdMem, dataIn, dataOut,
			partialSums, numBlocks);
}
template<typename T, typename Sum, typename Comm>
__global__ void priv_scan_kernel_28(const T* __restrict dataIn, T* dataOut,
		const int numBlocks, Comm* partialSums) {
	__shared__ T shrdMem[32];
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;

	priv_scan_stride_28<T, Sum, Comm, 28>(laneId, warpId, shrdMem, dataIn, dataOut,
			partialSums, numBlocks);
}
template<typename T, typename Sum, typename Comm>
__global__ void priv_scan_kernel_32(const T* __restrict dataIn, T* dataOut,
		const int numBlocks, Comm* partialSums) {
	__shared__ T shrdMem[32];
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;

	priv_scan_stride_32<T, Sum, Comm, 32>(laneId, warpId, shrdMem, dataIn, dataOut,
			partialSums, numBlocks);
}

template<typename T, typename Sum, typename Comm>
__global__ void priv_scan_kernel_36(const T* __restrict dataIn, T* dataOut,
		const int numBlocks, Comm* partialSums) {
	__shared__ T shrdMem[32];
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;

	priv_scan_stride_36<T, Sum, Comm, 36>(laneId, warpId, shrdMem, dataIn, dataOut,
			partialSums, numBlocks);
}
template<typename T, typename Sum, typename Comm>
__global__ void priv_scan_kernel_40(const T* __restrict dataIn, T* dataOut,
		const int numBlocks, Comm* partialSums) {
	__shared__ T shrdMem[32];
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;

	priv_scan_stride_40<T, Sum, Comm, 40>(laneId, warpId, shrdMem, dataIn, dataOut,
			partialSums, numBlocks);
}
template<typename T, typename Sum, typename Comm>
__global__ void priv_scan_kernel_44(const T* __restrict dataIn, T* dataOut,
		const int numBlocks, Comm* partialSums) {
	__shared__ T shrdMem[32];
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;

	priv_scan_stride_44<T, Sum, Comm, 44>(laneId, warpId, shrdMem, dataIn, dataOut,
			partialSums, numBlocks);
}
template<typename T, typename Sum, typename Comm>
__global__ void priv_scan_kernel_48(const T* __restrict dataIn, T* dataOut,
		const int numBlocks, Comm* partialSums) {
	__shared__ T shrdMem[32];
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;

	priv_scan_stride_48<T, Sum, Comm, 48>(laneId, warpId, shrdMem, dataIn, dataOut,
			partialSums, numBlocks);
}

template<typename T> int get_num_elements_per_thread() {
	return (sizeof(T) == 4) ? 44 : 20;
}

template<typename T, typename Sum, typename Comm, int ELEMENTS_PER_THREAD>
void scan(const int gridSize, const int blockSize, const int numShrdMemElements,
		const T* dataIn, T* dataOut, const int numBlocks, Comm* partialSums) {

	/*initialize the flag*/
	cudaMemset(partialSums, 0, numBlocks * sizeof(Comm));

	/*invoke the kernel*/
	switch (ELEMENTS_PER_THREAD) {
	case 4:
		priv_scan_kernel_4<T, Sum, Comm> <<<gridSize, blockSize>>>(dataIn, dataOut,
				numBlocks, partialSums);
		break;
	case 8:
		priv_scan_kernel_8<T, Sum, Comm> <<<gridSize, blockSize>>>(dataIn, dataOut,
				numBlocks, partialSums);
		break;
	case 16:
		priv_scan_kernel_16<T, Sum, Comm> <<<gridSize, blockSize>>>(dataIn, dataOut,
				numBlocks, partialSums);
		break;
	case 20:
		priv_scan_kernel_20<T, Sum, Comm> <<<gridSize, blockSize>>>(dataIn, dataOut,
				numBlocks, partialSums);
		break;
	case 24:
		priv_scan_kernel_24<T, Sum, Comm> <<<gridSize, blockSize>>>(dataIn, dataOut,
				numBlocks, partialSums);
		break;
	case 28:
		priv_scan_kernel_28<T, Sum, Comm> <<<gridSize, blockSize>>>(dataIn, dataOut,
				numBlocks, partialSums);
		break;
	case 32:
		priv_scan_kernel_32<T, Sum, Comm> <<<gridSize, blockSize>>>(dataIn, dataOut,
				numBlocks, partialSums);
		break;
	case 36:
		priv_scan_kernel_36<T, Sum, Comm> <<<gridSize, blockSize>>>(dataIn, dataOut,
				numBlocks, partialSums);
		break;
	case 40:
		priv_scan_kernel_40<T, Sum, Comm> <<<gridSize, blockSize>>>(dataIn, dataOut,
				numBlocks, partialSums);
		break;
	case 44:
		priv_scan_kernel_44<T, Sum, Comm> <<<gridSize, blockSize>>>(dataIn, dataOut,
				numBlocks, partialSums);
		break;
	case 48:
		priv_scan_kernel_48<T, Sum, Comm> <<<gridSize, blockSize>>>(dataIn, dataOut,
				numBlocks, partialSums);
		break;
	}

	/*synchronize the kernel*/
	cudaDeviceSynchronize();
}

} /*Scan namespace*/

#endif /* SCAN_CUH_ */
