#include <stdio.h>
#include <iostream>
#include "integralImage.h"


int main(int argc, char** argv)
{
	//using namespace BlockScan;
	extern int mainLF(int argc, char** argv);
	//mainLF(argc, argv);

	//Test();

	extern void TestBlockScan();
	TestBlockScan();

	extern void TestSerielScan();
	TestSerielScan();

	return 0;
}