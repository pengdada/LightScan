/*
 * Options.h
 *
 *  Created on: Aug 20, 2015
 *      Author: Yongchao Liu
 *      Affiliation: School of Computational Science & Engineering, Georgia Institute of Technology
 *      Email: yliu@cc.gatech.edu
 *      
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
using namespace std;

#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError(const char* file, const int32_t line) {
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		cerr << "cudaCheckError() failed at " << file << ":" << line << " : "
				<< cudaGetErrorString(err) << endl;
		exit(-1);
	}
}

struct Options {

	Options() {
		_verify = 0;
		_numElemsPerThread = 0;
		_numElements = 1 << 25;
		_maxIters = 1;
		_gpuIndex = 0;
	}

	/*parse parameter*/
	bool parse(int argc, char* argv[]);
	void printUsage();

	/*member variables*/
	int _verify;
	int _numElemsPerThread;
	int _numElements;
	int _maxIters;
	int _gpuIndex;
	vector<pair<int, cudaDeviceProp> > _deviceProps;
};

#ifdef _WIN32
#include <stdlib.h>
inline long long random() {
	return rand();
}
#else

#endif

#endif /* OPTIONS_H_ */
