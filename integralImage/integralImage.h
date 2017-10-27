#ifndef __INTEGRAL_IMAGE_H
#define __INTEGRAL_IMAGE_H
#include <vector>
#include <memory>

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
			*p1 += *p2 + sum;
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

static void Test() {
	int width = 8;
	int height = 8;
	int size = width*height;
	std::vector<int> src(size), dst1(size), dst2(size);
	std::fill(src.begin(), src.end(), 1);

	IntegralImageSerial(&src[0], &dst1[0], width, height);
	IntegralImageParallel(&src[0], &dst2[0], width, height);

	if (memcmp(&dst1[0], &dst2[0], size*sizeof(dst2[0])) == 0) {
		printf("memcmp is ok\n");
	}
	else {
		printf("memcmp is failed\n");
	}

}













#endif //__INTEGRAL_IMAGE_H