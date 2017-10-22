/*
 * LightScan.cu
 *
 *  Created on: Aug 18, 2015
 *      Author: Yongchao Liu
 *		Affiliation: Gerogia Institute of Technology
 *		Official Homepage: http://www.cc.gatech.edu/~yliu
 *		Personal Homepage: https://sites.google.com/site/yongchaosoftware
 *      
 */
#include "Options.h"
#include "Scan.cuh"
#include "Operator.cuh"

template<typename T>
bool CPUverify(T *hostData, T *hostResult, int num, int maxIters) {
	// cpu verify
	for (int iter = 0; iter < maxIters; ++iter) {
		for (int i = 0; i < num - 1; i++) {
			hostData[i + 1] = hostData[i] + hostData[i + 1];
		}
	}

	T diff = 0;
	for (int i = 0; i < num; i++) {
		diff += hostData[i] - hostResult[i];
	}
	cout << "CPU verify result diff (GPUvsCPU) = " << diff << endl;

	bool bTestResult = false;
	if (diff == 0) {
		bTestResult = true;
	}
	return bTestResult;
}

template<typename T>
bool parallel_scan(Options& opt) {
	T *hostData, *hostResult;
	T *devData;
	utils::CommPair<T> *devPartialSums;

	const int numElements = opt._numElements;
	const cudaDeviceProp& deviceProp = opt._deviceProps[opt._gpuIndex].second;

	/*set GPU*/
	cout << "Use GPU: " << opt._gpuIndex << endl;
	cudaSetDevice(opt._deviceProps[opt._gpuIndex].first);
	CudaCheckError();

	/*host-side memory allocation*/
	cudaMallocHost((void **) &hostData, sizeof(T) * numElements);
	CudaCheckError();

	cudaMallocHost((void **) &hostResult, sizeof(T) * numElements);
	CudaCheckError();

	//initialize data:
	cout << "Scan using cyclic-based approach" << endl;
	cout << "---------------------------------------------------" << endl;
	cout << "Initialize test data [1, 1, 1...] of " << numElements
			<< " elements" << endl;
	for (int i = 0; i < numElements; i++) {
		//hostData[i] = random() & 0xFF;
		hostData[i] = 1;
	}
	/*kernel configuration*/
	const int numElementsPerThread =
			opt._numElemsPerThread > 0 ?
					opt._numElemsPerThread :
					Scan::get_num_elements_per_thread<T>();
	const int numThreadsPerBlock = deviceProp.maxThreadsPerBlock/4; /*use the maximum number of threads per block*/
	const int numElementsPerBlock = numThreadsPerBlock * numElementsPerThread; /*number of elements per thread block*/
	const int numElementsAligned = (numElements + numElementsPerBlock - 1)
			/ numElementsPerBlock * numElementsPerBlock;
	const int numBlocksPerGrid = deviceProp.multiProcessorCount;

	cout << "numElementsPerThread: " << numElementsPerThread << endl;
	cout << "numThreadsPerBlock: " << numThreadsPerBlock << endl;
	cout << "numBlocksPerGrid: " << numBlocksPerGrid << endl;
	cout << "numElementsAligned: " << numElementsAligned << endl;
	cout << "Number of iterations: " << opt._maxIters << endl;

	// initialize a timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	CudaCheckError();

	cudaEventCreate(&stop);
	CudaCheckError();

	float et = 0, inc = 0;
	cudaMalloc((void **) &devData, numElementsAligned * sizeof(T));
	CudaCheckError();

	cudaMalloc((void **) &devPartialSums,
			numElementsAligned / numElementsPerBlock
					* sizeof(utils::CommPair<T>));
	CudaCheckError();

	/*transfer data to the device*/
	cudaMemcpy(devData, hostData, numElements * sizeof(T),
			cudaMemcpyHostToDevice);
	CudaCheckError();

	cout << "Invoke the kernel" << endl;
	/*start recording the runtime*/
	cudaEventRecord(start, 0);
	CudaCheckError();

	/*define an operator for the Scan operation*/
	typedef scanop::Add<T> SUM;
	
	/*invoke the kernel*/
	for (int i = 0; i < opt._maxIters; ++i) {
		switch (numElementsPerThread) {
		case 4:
			Scan::scan<T, SUM, utils::CommPair<T>, 4>(numBlocksPerGrid,
					numThreadsPerBlock, numThreadsPerBlock, devData, devData,
					numElementsAligned / numElementsPerBlock, devPartialSums);
			break;
		case 8:
			Scan::scan<T, SUM, utils::CommPair<T>, 8>(numBlocksPerGrid,
					numThreadsPerBlock, numThreadsPerBlock, devData, devData,
					numElementsAligned / numElementsPerBlock, devPartialSums);
			break;
		case 16:
			Scan::scan<T, SUM, utils::CommPair<T>, 16>(numBlocksPerGrid,
					numThreadsPerBlock, numThreadsPerBlock, devData, devData,
					numElementsAligned / numElementsPerBlock, devPartialSums);
			break;
		case 20:
			Scan::scan<T, SUM, utils::CommPair<T>, 20>(numBlocksPerGrid,
					numThreadsPerBlock, numThreadsPerBlock, devData, devData,
					numElementsAligned / numElementsPerBlock, devPartialSums);
			break;
		case 24:
			Scan::scan<T, SUM, utils::CommPair<T>, 24>(numBlocksPerGrid,
					numThreadsPerBlock, numThreadsPerBlock, devData, devData,
					numElementsAligned / numElementsPerBlock, devPartialSums);
			break;
		case 28:
			Scan::scan<T, SUM, utils::CommPair<T>, 28>(numBlocksPerGrid,
					numThreadsPerBlock, numThreadsPerBlock, devData, devData,
					numElementsAligned / numElementsPerBlock, devPartialSums);
			break;
		case 32:
			Scan::scan<T, SUM, utils::CommPair<T>, 32>(numBlocksPerGrid,
					numThreadsPerBlock, numThreadsPerBlock, devData, devData,
					numElementsAligned / numElementsPerBlock, devPartialSums);
			break;
		case 36:
			Scan::scan<T, SUM, utils::CommPair<T>, 36>(numBlocksPerGrid,
					numThreadsPerBlock, numThreadsPerBlock, devData, devData,
					numElementsAligned / numElementsPerBlock, devPartialSums);
			break;
		case 40:
			Scan::scan<T, SUM, utils::CommPair<T>, 40>(numBlocksPerGrid,
					numThreadsPerBlock, numThreadsPerBlock, devData, devData,
					numElementsAligned / numElementsPerBlock, devPartialSums);
			break;
		case 44:
			Scan::scan<T, SUM, utils::CommPair<T>, 44>(numBlocksPerGrid,
					numThreadsPerBlock, numThreadsPerBlock, devData, devData,
					numElementsAligned / numElementsPerBlock, devPartialSums);
			break;
		case 48:
			Scan::scan<T, SUM, utils::CommPair<T>, 48>(numBlocksPerGrid,
					numThreadsPerBlock, numThreadsPerBlock, devData, devData,
					numElementsAligned / numElementsPerBlock, devPartialSums);
			break;
		default:
			cerr << "Unsupported number of elements per thread: "
					<< numElementsPerThread << endl;
			exit(-1);
		}
	}

	/*end recording the runtime*/
	cudaEventRecord(stop, 0);
	CudaCheckError();

	cudaEventSynchronize(stop);
	CudaCheckError();

	cudaEventElapsedTime(&inc, start, stop);
	CudaCheckError();
	cout << "Finished the kernel" << endl;

	et += inc;
	et /= opt._maxIters;

	/*load back the data*/
	cudaMemcpy(hostResult, devData, numElements * sizeof(T),
			cudaMemcpyDeviceToHost);
	CudaCheckError();

	printf("Time (ms): %f\n", et);
	printf("%d elements scanned in %f ms -> %f (aligned %f) GigaElements/s\n",
			numElements, et, numElements / (et / 1000.0f) / 1000000000.0f,
			numElementsAligned / (et / 1000.0f) / 1000000000.0f);

	/*verify the results*/
	bool bTestResult = opt._verify ? 
					CPUverify<T>(hostData, hostResult, numElements,
							opt._maxIters) : true;

	cudaFreeHost(hostData);
	CudaCheckError();

	cudaFreeHost(hostResult);
	CudaCheckError();

	cudaFree(devData);
	CudaCheckError();

	cudaFree(devPartialSums);
	CudaCheckError();

	return bTestResult;
}

int main(int argc, char* argv[]) {
	Options opt;

	/*parse parameters*/
	if (opt.parse(argc, argv) == false) {
		return -1;
	}

	/*invoke the kernel*/
	bool ret = parallel_scan<int>(opt);
	if (opt._verify) {
		if (ret) {
			cout << "The scan results are correct" << endl;
		} else {
			cout << "The scan results are incorrect" << endl;
		}
	}
	return 0;
}

