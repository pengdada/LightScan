/*
 * Operator.cuh
 *
 *  Created on: Aug 27, 2015
 *      Author: Yongchao Liu
 *		Affiliation: Gerogia Institute of Technology
 *		Official Homepage: http://www.cc.gatech.edu/~yliu
 *		Personal Homepage: https://sites.google.com/site/yongchaosoftware
 */

#ifndef OPERATOR_CUH_
#define OPERATOR_CUH_

namespace scanop {
template<typename T>
struct Add{
	__host__  __device__  __forceinline__ T operator()(const T& a, const T& b) {
		return a + b;
	}
};

/***********************************
 * Max operation
 */
template<typename T> struct Max;
template<> struct Max<int> {
	__host__ __device__ __forceinline__ int operator()(int a, int b) {
		return max(a, b);
	}
};
template<> struct Max<unsigned int> {
	__host__ __device__ __forceinline__ unsigned int operator()(unsigned int a, unsigned int b) {
		return max(a, b);
	}
};
template<> struct Max<long long> {
	__host__ __device__ __forceinline__ long long operator()(long long a, long long b) {
		return max(a, b);
	}
};
template<> struct Max<unsigned long long> {
	__host__ __device__ __forceinline__ unsigned long long operator()(unsigned long long a, unsigned long long b) {
		return max(a, b);
	}
};
template<> struct Max<float> {
	__host__ __device__ __forceinline__ float operator()(float a, float b) {
		return fmax(a, b);
	}
};
template<> struct Max<double> {
	__host__ __device__ __forceinline__ double operator()(double a, double b) {
		return fmax(a, b);
	}
};
/***********************************
 * Min operation
 */
template<typename T> struct Min;
template<> struct Min<int> {
	__host__ __device__ __forceinline__ int operator()(int a, int b) {
		return min(a, b);
	}
};
template<> struct Min<unsigned int> {
	__host__ __device__ __forceinline__ unsigned int operator()(unsigned int a, unsigned int b) {
		return min(a, b);
	}
};
template<> struct Min<long long> {
	__host__ __device__ __forceinline__ long long operator()(long long a, long long b) {
		return min(a, b);
	}
};
template<> struct Min<unsigned long long> {
	__host__ __device__ __forceinline__ unsigned long long operator()(unsigned long long a, unsigned long long b) {
		return min(a, b);
	}
};
template<> struct Min<float> {
	__host__ __device__ __forceinline__ float operator()(float a, float b) {
		return fmin(a, b);
	}
};
template<> struct Min<double> {
	__host__ __device__ __forceinline__ double operator()(double a, double b) {
		return fmin(a, b);
	}
};

}/*operator namespace*/

#endif /* OPERATOR_CUH_ */
