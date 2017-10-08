#include "Kernel.cuh"
#include <cmath>

typedef struct RGB {
	uint8_t r;
	uint8_t g; 
	uint8_t b; 
};

RGB int_to_rgb(uint32_t pixel , bool isBigEndian) {
	RGB rgb; 
	
}


__device__
void initialize_2D_array(uint32_t *array, int size_w, int size_h) {
	int i = blockIdx.x;
	int j = threadIdx.x;

	array[i*size_w + j] = 0;
}


__global__
void GPU_compute_greyscale_luminance(uint32_t *array, int size_w, int size_h) {

}


__global__
void GPU_compute_hmap(uint32_t *array, int size_w, int size_h) {
	initialize_2D_array(array, size_w, size_h); 
	

}




 uint32_t* GPU_Initialize(int w, int h) {
	uint32_t * array, *D_array;
	array = new uint32_t[h*w];



	cudaMalloc((void**)&D_array, w*h * sizeof(uint32_t));
	cudaMemcpy(D_array, array, w*h * sizeof(uint32_t), cudaMemcpyHostToDevice);


	int thread_per_blocks = 512;
	int blocks_per_grid = floor((w*h + thread_per_blocks - 1) / thread_per_blocks)+1;

	GPU_compute_hmap << <blocks_per_grid , thread_per_blocks >> > (D_array, w, h);
	cudaError_t err = cudaGetLastError(); 
	if (!err != cudaSuccess) {
		printf("\nError : %s \n", cudaGetErrorString(err)); 
	}
	cudaMemcpy(array, D_array, w*h * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaFree(D_array);

	return array;
}



 uint32_t* GPU_compute_greyscale(uint32_t* array, int width, int height) {

 }