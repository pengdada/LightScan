#include <stdio.h>
#include <iostream>


int main(int argc, char** argv)
{
	extern int mainLF(int argc, char** argv);
	mainLF(argc, argv);
	return 0;
}