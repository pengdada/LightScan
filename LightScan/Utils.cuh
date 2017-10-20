/*
 * Utils.cuh
 *
 *  Created on: Aug 25, 2015
 *      Author: Yongchao Liu
 *		Affiliation: Gerogia Institute of Technology
 *		Official Homepage: http://www.cc.gatech.edu/~yliu
 *		Personal Homepage: https://sites.google.com/site/yongchaosoftware
 */

#ifndef UTILS_CUH_
#define UTILS_CUH_
#include <stdint.h>

namespace utils {
/*pair data for communication*/
template<typename T> class CommPair;

template<> class CommPair<int> {
public:
	__device__ __host__ __forceinline__ CommPair<int>(int a, int b) :
			first(a), second(b) {
	}
	int first; /*flag*/
	int second; /*data to be communicated*/
}
#ifdef _WIN32
;
#else
__attribute__((packed, aligned(8)));
#endif

template<> class CommPair<unsigned int> {
public:
	__device__ __host__ __forceinline__ CommPair<unsigned int>(unsigned int a, unsigned int b) :
			first(a), second(b) {
	}
	unsigned int first;
	unsigned int second;
}
#ifdef _WIN32
;
#else
#else__attribute__((packed, aligned(8)));
#endif

template<> class CommPair<long long> {
public:
	__device__ __host__ __forceinline__ CommPair<long long>(long long  a, long long b) :
			first(a), second(b) {
	}
	long long first;
	long long second;
}
#ifdef _WIN32
;
#else
__attribute__((packed, aligned(8)));
#endif

template<> class CommPair<unsigned long long> {
public:
	__device__ __host__ __forceinline__ CommPair<unsigned long long>(unsigned long long a, unsigned long long b) :
			first(a), second(b) {
	}
	unsigned long long first;
	unsigned long long second;
}
#ifdef _WIN32
;
#else
__attribute__((packed, aligned(8)));
#endif

template<> class CommPair<float> {
public:
	__device__ __host__ __forceinline__ CommPair<float>(int a, float b) :
			first(a), second(b) {
	}
	int first;
	float second;
}
#ifdef _WIN32
;
#else
__attribute__((packed, aligned(8)));
#endif

template<> class CommPair<double> {
public:
	__device__ __host__ __forceinline__ CommPair<double>(long long a, double b) :
			first(a), second(b) {
	}
	long long first;
	double second;
}
#ifdef _WIN32
;
#else
__attribute__((packed, aligned(8)));
#endif

#if 0
template<typename T>
__device__ __forceinline__ T atomic_read(T* data) {
	if (sizeof(T) == 4) {
		unsigned int retval;
		unsigned int* ptr = reinterpret_cast<unsigned int*>(data);
		asm volatile ("ld.cg.u32 %0, [%1];" : "=r"(retval) : "l"(ptr));
		return *(reinterpret_cast<T*>(&retval));
	} else if (sizeof(T) == 8) {
		unsigned long long retval;
		unsigned long long*ptr = reinterpret_cast<unsigned long long*>(data);
		asm volatile ("ld.cg.u64 %0, [%1];" : "=l"(retval) : "l"(ptr));
		return *(reinterpret_cast<T*>(&retval));
	} else {
		ulonglong2 retval;
		ulonglong2* ptr = reinterpret_cast<ulonglong2*>(data);
		asm volatile("ld.cg.v2.u64 {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(ptr));
		return *(reinterpret_cast<T*>(&retval));
	}
}
#else
template<typename T>
__device__       __forceinline__ T atomic_read(T* data);

template<>
__device__       __forceinline__ CommPair<int> atomic_read(CommPair<int>* data) {
	unsigned long long retval;
	unsigned long long*ptr = reinterpret_cast<unsigned long long*>(data);
	asm volatile ("ld.cg.u64 %0, [%1];" : "=l"(retval) : "l"(ptr));
	return *(reinterpret_cast<CommPair<int>*>(&retval));
}
template<>
__device__       __forceinline__ CommPair<unsigned int> atomic_read(
		CommPair<unsigned int>* data) {
	unsigned long long retval;
	unsigned long long*ptr = reinterpret_cast<unsigned long long*>(data);
	asm volatile ("ld.cg.u64 %0, [%1];" : "=l"(retval) : "l"(ptr));
	return *(reinterpret_cast<CommPair<unsigned int>*>(&retval));
}
template<>
__device__       __forceinline__ CommPair<long long > atomic_read(
		CommPair<long long>* data) {
	ulonglong2 retval;
	ulonglong2* ptr = reinterpret_cast<ulonglong2*>(data);
	asm volatile("ld.cg.v2.u64 {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(ptr));
	return *(reinterpret_cast<CommPair<long long >*>(&retval));
}
template<>
__device__       __forceinline__ CommPair<unsigned long long> atomic_read(
		CommPair<unsigned long long>* data) {
	ulonglong2 retval;
	ulonglong2* ptr = reinterpret_cast<ulonglong2*>(data);
	asm volatile("ld.cg.v2.u64 {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(ptr));
	return *(reinterpret_cast<CommPair<unsigned long long>*>(&retval));
}

template<>
__device__       __forceinline__ CommPair<float> atomic_read(CommPair<float>* data) {
	unsigned long long retval;
	unsigned long long*ptr = reinterpret_cast<unsigned long long*>(data);
	asm volatile ("ld.cg.u64 %0, [%1];" : "=l"(retval) : "l"(ptr));
	return *(reinterpret_cast<CommPair<float>*>(&retval));
}
template<>
__device__       __forceinline__ CommPair<double> atomic_read(
		CommPair<double>* data) {
	ulonglong2 retval;
	ulonglong2* ptr = reinterpret_cast<ulonglong2*>(data);
	asm volatile("ld.cg.v2.u64 {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(ptr));
	return *(reinterpret_cast<CommPair<double>*>(&retval));
}
#endif

#if 0
template<typename T>
__device__ __forceinline__ void atomic_store(T* data,  T& val) {
	if (sizeof(T) == 4) {
		unsigned int* ptr = reinterpret_cast<unsigned int*>(data);
		unsigned int stval = *reinterpret_cast<unsigned int*>(&val);
		asm volatile ("st.cg.u32 [%0], %1;" : : "l"(ptr), "r"(stval));
	} else if (sizeof(T) == 8) {
		unsigned long long* ptr = reinterpret_cast<unsigned long long*>(data);
		unsigned long long stval = *reinterpret_cast<unsigned long long*>(&val);
		asm volatile ("st.cg.u64 [%0], %1;" : : "l"(ptr), "l"(stval));

	} else {
		ulonglong2 * ptr = reinterpret_cast<ulonglong2*>(data);
		ulonglong2 stval = *reinterpret_cast<ulonglong2*>(&val);
		asm volatile ("st.cg.v2.u64 [%0], {%1, %2};" : : "l"(ptr), "l"(stval.x), "l"(stval.y));
	}
}
#else
template<typename T>
__device__ __forceinline__ void atomic_store(T* data,  T val);

template<>
__device__ __forceinline__ void atomic_store(CommPair<int>* data,
		CommPair<int> val) {
	unsigned long long* ptr = reinterpret_cast<unsigned long long*>(data);
	unsigned long long stval = *reinterpret_cast<unsigned long long*>(&val);
	asm volatile ("st.cg.u64 [%0], %1;" : : "l"(ptr), "l"(stval));
}
template<>
__device__ __forceinline__ void atomic_store(CommPair<unsigned int>* data,
		 CommPair<unsigned int> val) {
	unsigned long long* ptr = reinterpret_cast<unsigned long long*>(data);
	unsigned long long stval = *reinterpret_cast<unsigned long long*>(&val);
	asm volatile ("st.cg.u64 [%0], %1;" : : "l"(ptr), "l"(stval));
}
template<>
__device__ __forceinline__ void atomic_store(CommPair<long long>* data,
		 CommPair<long long> val) {
	ulonglong2 * ptr = reinterpret_cast<ulonglong2*>(data);
	ulonglong2 stval = *reinterpret_cast<ulonglong2*>(&val);
	asm volatile ("st.cg.v2.u64 [%0], {%1, %2};" : : "l"(ptr), "l"(stval.x), "l"(stval.y));
}
template<>
__device__ __forceinline__ void atomic_store(CommPair<unsigned long long>* data,
		 CommPair<unsigned long long> val) {
	ulonglong2 * ptr = reinterpret_cast<ulonglong2*>(data);
	ulonglong2 stval = *reinterpret_cast<ulonglong2*>(&val);
	asm volatile ("st.cg.v2.u64 [%0], {%1, %2};" : : "l"(ptr), "l"(stval.x), "l"(stval.y));
}
template<>
__device__ __forceinline__ void atomic_store(CommPair<float>* data,
		 CommPair<float> val) {
	unsigned long long* ptr = reinterpret_cast<unsigned long long*>(data);
	unsigned long long stval = *reinterpret_cast<unsigned long long*>(&val);
	asm volatile ("st.cg.u64 [%0], %1;" : : "l"(ptr), "l"(stval));
}
template<>
__device__ __forceinline__ void atomic_store(CommPair<double>* data,
		 CommPair<double> val) {
	ulonglong2 * ptr = reinterpret_cast<ulonglong2*>(data);
	ulonglong2 stval = *reinterpret_cast<ulonglong2*>(&val);
	asm volatile ("st.cg.v2.u64 [%0], {%1, %2};" : : "l"(ptr), "l"(stval.x), "l"(stval.y));
}
#endif

template<typename T, typename Comm>
__device__   __forceinline__ T busy_wait_comm(Comm* buffer,  int gbid,
		 T partialSum) {
#if 1
	T val = 0;
	if (gbid > 0) {
		/*busy-wait*/
		Comm flag = utils::atomic_read<Comm>(buffer + gbid - 1);
		while (flag.first == 0) {
			flag = utils::atomic_read<Comm>(buffer + gbid - 1);
		}
		val = flag.second;
	}
	if (threadIdx.x == blockDim.x - 1) {
		/*atomically write the partial sum of the thread block to global memory*/
		utils::atomic_store<Comm>(buffer + gbid, Comm(1, partialSum + val));
	}
	/*synchronize all threads per thread block*/
	__syncthreads();

	return val;
#else
	return 0;
#endif
}
}/*utils namespace*/

#endif /* UTILS_CUH_ */
