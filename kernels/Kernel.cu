#include "Kernel.cuh"

__global__ void  GPU_compute_hmap(uint32_t* array, int width, int height);
__global__ void GPU_compute_nmap();
__global__ void GPU_compute_greyscale();
__global__ void GPU_set_greyscale();




__device__
void initialize_2D_array(uint32_t *array, int size_w, int size_h) {
	int i = blockIdx.x * threadIdx.x * threadIdx.y;
	array[i] = 0;
}





__global__
void GPU_compute_hmap(uint32_t *array, int size_w, int size_h) {
	initialize_2D_array(array, size_w, size_h);


}




 uint32_t* GPU_action(int w, int h) {
	uint32_t * array, *D_array;
	array = new uint32_t[h*w];



	cudaMalloc((void**)&D_array, w*h * sizeof(uint32_t));
	cudaMemcpy(D_array, array, w*h * sizeof(uint32_t), cudaMemcpyHostToDevice);


	int thread_per_blocks = 512;
	int blocks_per_grid = (w*h + thread_per_blocks - 1) / thread_per_blocks;

	GPU_compute_hmap << <blocks_per_grid, thread_per_blocks >> > (D_array, w, h);

	cudaMemcpy(array, D_array, w*h * sizeof(uint32_t), cudaMemcpyDeviceToHost);
	cudaFree(D_array);

	return array;
}