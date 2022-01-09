


/* My own implementation : Negative */
__global__ void alg1(byte* source, uint32_t filesize) {
	uint32_t id = 3 * ((blockIdx.x * THERADS) + threadIdx.x);
	if (id < filesize) {
		source[id] = 255 - source[id];
		source[id + 1] = 255 - source[id + 1];
		source[id + 2] = 255 - source[id + 2];
	}

}

/* My own implementation : Grayscale */
__global__ void alg2(byte* source, uint32_t filesize) {
	uint32_t id = 3 * ((blockIdx.x * THERADS) + threadIdx.x);
	if (id < filesize) {
		source[id] = 0.299 * source[id] + 0.587 * source[id + 1] + 0.114 * source[id + 2];
		source[id + 1] = source[id];
		source[id + 2] = source[id];
	}
}

/* My own implementation : Sepia */
__global__ void alg3(byte* source, uint32_t filesize) {
	uint32_t id = 3 * ((blockIdx.x * THERADS) + threadIdx.x);
	if (id < filesize) {
		source[id] = 0.272 * source[id + 2] + 0.534 * source[id + 1] + 0.131 * source[id];
		source[id + 1] = 0.349 * source[id + 2] + 0.686 * source[id + 1] + 0.168 * source[id];
		source[id + 2] = 0.393 * source[id + 2] + 0.769 * source[id + 1] + 0.189 * source[id];
	}
}

/* My own idea : somehow increses contrast? dunno, just a random ideea */
__global__ void alg4(byte* source, uint32_t filesize) {
	uint32_t id = 3 * ((blockIdx.x * THERADS) + threadIdx.x);
	if (id < filesize) {
		if(source[id] < 40)
			source[id] += 50;
		else
			source[id] += 20;
		if (source[id + 1] < 40)
			source[id + 1] += 50;
		else
			source[id + 1] += 20;
		if (source[id + 2] < 40)
			source[id + 2] += 50;
		else
			source[id + 2] += 20;
	}
}

/* My own idea : I think this loogs cool, since it makes
				everything more dramatic by decreasing the luminosity of every pixel */
__global__ void alg5(byte* source, uint32_t filesize) {
	uint32_t id = 3 * ((blockIdx.x * THERADS) + threadIdx.x);
	if (id < filesize) {
		if (source[id + 1] > 70) {
			if (source[id] < 60)
				source[id] = 0;
			source[id] -= 60;
		}
		if (source[id + 2] > 70) {
			if (source[id+1] < 60)
				source[id+1] = 0;
			source[id+1] -= 60;
		}
		if (source[id] > 70) {
			if (source[id+2] < 60)
				source[id+2] = 0;
			source[id+2] -= 60;
		}
	}
}

/* My own idea : This will create a fuzzy image.
				 It plays with bith by overflowing them in multiplication */
__global__ void alg6(byte* source, uint32_t filesize) {
	uint32_t id = 3 * ((blockIdx.x * THERADS) + threadIdx.x);
	if (id < filesize) {
		source[id] =  source[id] * source[id + 1];
		source[id + 1] = source[id + 1] * source[id + 2];
		source[id + 2] = source[id + 2] * source[id];
	}
}

/* My own idea : Same as alg6, but with adition */
__global__ void alg7(byte* source, uint32_t filesize) {
	uint32_t id = 3 * ((blockIdx.x * THERADS) + threadIdx.x);
	if (id < filesize) {
		source[id] = source[id] + source[id + 1];
		source[id + 1] = source[id + 1] + source[id + 2];
		source[id + 2] = source[id + 2] + source[id];
	}
}

/* My own idea : Swaping color between subpixels */
__global__ void alg8(byte* source, uint32_t filesize) {
	uint32_t id = 3 * ((blockIdx.x * THERADS) + threadIdx.x);
	byte temp = source[id];
	if (id < filesize) {
		source[id] = source[id + 1];
		source[id + 1] = source[id + 2];
		source[id + 2] = temp;
	}
}

/* This function start all the jobs for the GPU. Every algorith takes as parameter a memory location 
and the ammount of bytes to process, a Limit. The algorithms will start all at once.				*/
void start_processing() {
	alg1 << <blocks, threads >> > (alg1_input, fileSize - HEADER_SIZE);
	alg2 << <blocks, threads >> > (alg2_input, fileSize - HEADER_SIZE);
	alg3 << <blocks, threads >> > (alg3_input, fileSize - HEADER_SIZE);
	alg4 << <blocks, threads >> > (alg4_input, fileSize - HEADER_SIZE);
	alg5 << <blocks, threads >> > (alg5_input, fileSize - HEADER_SIZE);
	alg6 << <blocks, threads >> > (alg6_input, fileSize - HEADER_SIZE);
	alg7 << <blocks, threads >> > (alg7_input, fileSize - HEADER_SIZE);
	alg8 << <blocks, threads >> > (alg8_input, fileSize - HEADER_SIZE);
}

/* After the processing, all the processing is saved to RAM in order to be saved to storage device */
void save_processing() {

	cout << "Copying filtered images to RAM..." << endl;
	cudaMemcpy(alg1_output, alg1_input, (fileSize - HEADER_SIZE), cudaMemcpyDeviceToHost);
	cudaMemcpy(alg2_output, alg2_input, (fileSize - HEADER_SIZE), cudaMemcpyDeviceToHost);
	cudaMemcpy(alg3_output, alg3_input, (fileSize - HEADER_SIZE), cudaMemcpyDeviceToHost);
	cudaMemcpy(alg4_output, alg4_input, (fileSize - HEADER_SIZE), cudaMemcpyDeviceToHost);
	cudaMemcpy(alg5_output, alg5_input, (fileSize - HEADER_SIZE), cudaMemcpyDeviceToHost);
	cudaMemcpy(alg6_output, alg6_input, (fileSize - HEADER_SIZE), cudaMemcpyDeviceToHost);
	cudaMemcpy(alg7_output, alg7_input, (fileSize - HEADER_SIZE), cudaMemcpyDeviceToHost);
	cudaMemcpy(alg8_output, alg8_input, (fileSize - HEADER_SIZE), cudaMemcpyDeviceToHost);
	cout << "Writing files to storage device..." << endl;
	writeFile(alg1_output, "alg1.bmp");
	writeFile(alg2_output, "alg2.bmp");
	writeFile(alg3_output, "alg3.bmp");
	writeFile(alg4_output, "alg4.bmp");
	writeFile(alg5_output, "alg5.bmp");
	writeFile(alg6_output, "alg6.bmp");
	writeFile(alg7_output, "alg7.bmp");
	writeFile(alg8_output, "alg8.bmp");
}