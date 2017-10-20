/*
 * Options.cu
 *
 *  Created on: Aug 20, 2015
 *      Author: Yongchao Liu
 *      Affiliation: School of Computational Science & Engineering, Georgia Institute of Technology
 *      Email: yliu@cc.gatech.edu
 *      
 */
#include "Options.h"
#ifdef _WIN32
#include "wingetopt.h"
#else
#include "unistd.h"
#endif

void Options::printUsage() {
	printf("lightscan [options] number_of_elements\n");
	printf("Options:\n");
	printf("\t-n (number of elements per thread, default=%d)\n", _numElemsPerThread);
	printf("\t-i (number of iterations, default=%d)\n", _maxIters);
	printf("\t-g (GPU index, default=%d)\n", _gpuIndex);
	printf("\t-v (verify the results, default=%d)\n", _verify);
	printf("\n");
}
bool Options::parse(int argc, char* argv[]) {
	int num;
	int c;
	cudaDeviceProp prop;

	/*check parameters*/
	if (argc < 2) {
		printUsage();
		return false;
	}

	/*get GPU device*/
	cudaGetDeviceCount(&num);
	if (num == 0) {
		printf("No CUDA-enabled GPU is detected in the host\n");
		return false;
	}

	for (int i = 0; i < num; ++i) {

		/*get the device property*/
		cudaGetDeviceProperties(&prop, i);
		CudaCheckError();

		/*check the compute capability*/
		if (prop.major >= 3) {
			/*save the device*/
			_deviceProps.push_back(make_pair(i, prop));
			printf(
					"(%ld): Detected Compute SM %d.%d hardware with %d multi-processors\n",
					_deviceProps.size() - 1, prop.major, prop.minor,
					prop.multiProcessorCount);
		}
	}
	if (_deviceProps.size() == 0) {
		printf("No CUDA-enabled GPU with compute capability >= 3 available\n");
		return false;
	}

	while ((c = getopt(argc, argv, "n:i:g:v:")) != -1) {
		switch (c) {
		case 'n':
			_numElemsPerThread = atoi(optarg);
			if (_numElemsPerThread < 1) {
				_numElemsPerThread = 0;
			}
			break;
		case 'i':
			_maxIters = atoi(optarg);
			if (_maxIters < 1) {
				_maxIters = 1;
			}
			break;
		case 'g':
			_gpuIndex = atoi(optarg);
			if (_gpuIndex >= (int) _deviceProps.size()) {
				_gpuIndex = (int) _deviceProps.size() - 1;
			}
			if (_gpuIndex < 0) {
				_gpuIndex = 0;
			}
			break;
		case 'v':
			_verify = atoi(optarg);
			if (_verify) {
				_verify = 1;
			}
			break;
		default:
			printf("Unknown option: %s\n", optarg);
			break;
		}
	}
	/*get the elements per thread*/
	if (optind < argc) {
		_numElements = atoi(argv[optind]);
	} else {
		cout << "The number of elements must be specified" << endl;
		return false;
	}
	if (_numElements < 1) {
		_numElements = 1;
	}

	return true;
}

