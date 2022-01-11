#pragma once

#include <iostream>
#include <cstdint>

/*		-------GLOBAL VARIABLES-------		*/

using namespace std;

#define NOT_OK 0
#define DEBUG 0

typedef uint8_t byte;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef const char * String;



const uint8 HEADER_SIZE = 54;
byte inputHeader[54] = { 0 };

/* The path of the source original picture */
String inputPath = "C:\\Users\\peria\\Desktop\\Test.bmp";
String outputPath = "C:\\Users\\peria\\Desktop\\";

FILE* inputFile;

byte* pictureData;

const uint16 threads = 512;
uint32 fileSize = 0;
uint32 blocks = 0;

byte* alg1_input;
byte* alg2_input;
byte* alg3_input;
byte* alg4_input;
byte* alg5_input;
byte* alg6_input;
byte* alg7_input;
byte* alg8_input;

byte* alg1_output;
byte* alg2_output;
byte* alg3_output;
byte* alg4_output;
byte* alg5_output;
byte* alg6_output;
byte* alg7_output;
byte* alg8_output;

__device__ uint16 THREADS = threads;

/*		-------GLOBAL VARIABLES-------		*/

/* This function is used for debuging */
#if DEGUB
	void printFirst(uint64 stop, byte* pointer) {
		uint8 col = 0;
		for (uint64 i = 0; i < stop;i++) {
			cout << (int)pointer[i]<<" ";
			col++;
			if (col > 9) {
				col = 0;
				cout << endl;
			}
		}
	}
#endif

/* Calculates the minimum ammount of block necesary to fit the whole picture data */
void calculateBlocks() {
	blocks = ((float)((fileSize - HEADER_SIZE) / 3) / (float)threads)+1;
}

/* Reads the header of the original file and assigns the global variables */
void readHeader() {
	fread(inputHeader, sizeof(byte), HEADER_SIZE, inputFile);
	fileSize = *((uint64*)(&inputHeader[2]));
	pictureData = (byte*)malloc(fileSize - HEADER_SIZE);
}

/* Copies to ram all the memory locations needed for algorithms outputs and 
	reads the original file and copies it into the RAM						*/
void copyToRam() {
	fread(pictureData, sizeof(byte), fileSize - HEADER_SIZE, inputFile);
#if DEBUG
	printFirst(50, pictureData);
#endif
	cout << "The size of the file is: " << fileSize << " bytes" << endl;
	alg1_output = (byte*)malloc(fileSize - HEADER_SIZE);
	alg2_output = (byte*)malloc(fileSize - HEADER_SIZE);
	alg3_output = (byte*)malloc(fileSize - HEADER_SIZE);
	alg4_output = (byte*)malloc(fileSize - HEADER_SIZE);
	alg5_output = (byte*)malloc(fileSize - HEADER_SIZE);
	alg6_output = (byte*)malloc(fileSize - HEADER_SIZE);
	alg7_output = (byte*)malloc(fileSize - HEADER_SIZE);
	alg8_output = (byte*)malloc(fileSize - HEADER_SIZE);
}

/* Take as a parameter procesed pictures pixels and saves the 
	file by adding a header and the specified name			*/
void writeFile(byte* toWrite,String name) {
	int nameLength = strlen(name);
	int outputPathLength = strlen(outputPath);
	char filename[512] = {0};
	for (int i = 0; i < nameLength+outputPathLength; i++) {
		if(i<outputPathLength)
			filename[i] = outputPath[i];
		else 
			filename[i] = name[i-outputPathLength];
	}

	FILE* output = fopen(filename, "wb+");
	fwrite(inputHeader, 1, HEADER_SIZE, output);
	fwrite(toWrite, 1, fileSize-HEADER_SIZE, output);
	fclose(output);
}

/* Clears the RAM and VRAM after all the processing is complete */
void freeMemory() {
	cout << "Clearing RAM..." << endl;
	free(pictureData);
	free(alg1_output);
	free(alg2_output);
	free(alg3_output);
	free(alg4_output);
	free(alg5_output);
	free(alg6_output);
	free(alg7_output);
	free(alg8_output);
	cout << "Clearing VRAM..." << endl;
	cudaFree(alg1_input);
	cudaFree(alg2_input);
	cudaFree(alg3_input);
	cudaFree(alg4_input);
	cudaFree(alg5_input);
	cudaFree(alg6_input);
	cudaFree(alg7_input);
	cudaFree(alg8_input);
}

/* Alocate memory on the GPU for every algorithm and copies the original 
	image to every algorithm dedicated VRAM								*/
void copyToGpu() {

	cout << "Alocate memory on VRAM..." << endl;

	cudaMalloc(&alg1_input, (fileSize - HEADER_SIZE));
	cudaMalloc(&alg2_input, fileSize - HEADER_SIZE);
	cudaMalloc(&alg3_input, fileSize - HEADER_SIZE);
	cudaMalloc(&alg4_input, fileSize - HEADER_SIZE);
	cudaMalloc(&alg5_input, fileSize - HEADER_SIZE);
	cudaMalloc(&alg6_input, fileSize - HEADER_SIZE);
	cudaMalloc(&alg7_input, fileSize - HEADER_SIZE);
	cudaMalloc(&alg8_input, fileSize - HEADER_SIZE);

	cout << "Copying original image to VRAM..." << endl;

	cudaMemcpy(alg1_input, pictureData, (fileSize - HEADER_SIZE), cudaMemcpyHostToDevice);
	cudaMemcpy(alg2_input, pictureData, fileSize - HEADER_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(alg3_input, pictureData, fileSize - HEADER_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(alg4_input, pictureData, fileSize - HEADER_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(alg5_input, pictureData, fileSize - HEADER_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(alg6_input, pictureData, fileSize - HEADER_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(alg7_input, pictureData, fileSize - HEADER_SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(alg8_input, pictureData, fileSize - HEADER_SIZE, cudaMemcpyHostToDevice);
}

/* This initializes every global variable and calls the requred functions 
	in order for algorithms to be able to launch corectly					*/

void Init() {
	inputFile = fopen(inputPath, "rb");
	readHeader();
	calculateBlocks();
	if (pictureData != NOT_OK) {
		copyToRam();
		copyToGpu();
	}
	else {
		cout << "Error alocating memory for picture!" << endl;
	}
}


/* This end operations ensures no memory leakage is present and prints the end of the program */
void End() {
	fclose(inputFile);
	cout << "Input file closed..." << endl;
	cout << "Clearing memory..." << endl;
	freeMemory();
	cout << "Exiting...";
}









