#ifndef __INTEGRAL_IMAGE_H
#define __INTEGRAL_IMAGE_H

#include <vector>
#include <memory>
#include "cudaLib.cuh"
#include <iostream>
#include <chrono>

#ifdef _WIN32
#include <Windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#endif

template<typename T> inline
void IntegralRow(const T* src, T* dst, int count) {
	T sum = 0;
	for (int i = 0; i < count; i++) {
		sum += src[i];
		dst[i] = sum;
	}
}

template<typename T> inline
void IntegralStride(const T* src, int nStrideSrc, T* dst, int nStrideDst, int count) {
	T sum = 0;
	const T* p0 = src;
	T* p1 = dst;
	for (int i = 0; i < count; i++, p0 += nStrideSrc, p1 += nStrideDst) {
		sum += *p0;
		*p1 = sum;
	}
}

template<typename T> inline
void IntegralImageSerial(const T* src, T* dst, int width, int height) {
	T sum = 0;

	const T* p0 = src;
	T* p1 = dst;
	T* p2 = dst;
	for (int j = 0; j < width; j++, p0 ++, p1 ++) {
		sum += *p0;
		*p1 = sum;
	}
	for (int i = 1; i < height; i++) {
		sum = 0;
		for (int j = 0; j < width; j++, p0 ++, p1 ++, p2 ++) {
			sum += *p0;
			*p1 = *p2 + sum;
		}
	}
}

template<typename T> inline
void IntegralImageParallel(const T* src, T* dst, int width, int height) {
	T sum = 0;

	const T* p0 = src;
	T* p1 = dst;

#pragma omp parallel for
	for (int i = 0; i < height; i++) {
		IntegralRow(src + i*width, dst + i*width, width);
	}

#pragma omp parallel for
	for (int i = 0; i < width; i++) {
		IntegralStride(dst + i, width, dst + i, width, height);
	}
}

struct Timer {
	float elapsed;
	cudaEvent_t m_start_event;
	cudaEvent_t m_stop_event;
	Timer(){
		elapsed = 0.f;
		cudaEventCreate(&m_start_event);
		cudaEventCreate(&m_stop_event);
	}
	~Timer()
	{
		cudaEventDestroy(m_start_event);
		cudaEventDestroy(m_stop_event);
	}
	void start()
	{
		cudaEventRecord(m_start_event, 0);
	}
	void stop()
	{
		cudaEventRecord(m_stop_event, 0);
		cudaEventSynchronize(m_stop_event);
		cudaEventElapsedTime(&elapsed, m_start_event, m_stop_event);
	}
	float elapsedInMs() const
	{
		return elapsed;
	}
};

inline float getTime() {
#ifdef _WIN32
	return timeGetTime();
#endif
}

template<typename T> inline
float GetAvgTime(const T* src, T* dst, int width, int height, int type) {
	const int N = 6;
	auto start = std::chrono::system_clock::now();
	for (int i = 0; i < N; i++) {
		if (type == 0) IntegralImageSerial(src, dst, width, height);
		if (type == 1) IntegralImageParallel(src, dst, width, height);
	}
	auto end = std::chrono::system_clock::now();
	auto diff = end - start;
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(diff).count() / 1000000.f;
	return duration / N;
}

static void TestCPU(int width, int height) {
	//int width = 1024*1;
	//int height = 1024*2;
	int size = width*height;
	std::vector<int> src(size), dst1(size), dst2(size);
	std::fill(src.begin(), src.end(), 1);
	{
		float tm1 = GetAvgTime(&src[0], &dst1[0], width, height, 0);
		float tm2 = GetAvgTime(&src[0], &dst2[0], width, height, 1);
		printf("%d, %d, IntegralImageSerial tm1 = %f \n    IntegralImageParallel tm2 = %f\n", width, height, tm1, tm2);

		FILE* flog = fopen("d:/log.txt", "at");
		if(flog){
			fprintf(flog, "%f ", tm1);
			fclose(flog);
		}
	}

	if (memcmp(&dst1[0], &dst2[0], size*sizeof(dst2[0])) == 0) {
		printf("memcmp is ok\n");
	}
	else {
		printf("memcmp is failed\n");
	}
}

static void Test() {
	for(int i = 1; i < 10; i++){
		TestCPU(i*1024, i*1024);
	}
}


#endif //__INTEGRAL_IMAGE_H