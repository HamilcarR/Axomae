#include "Kernel.cuh"
#include <cmath>

typedef struct RGB {
	uint8_t r;
	uint8_t g; 
	uint8_t b; 
	uint8_t a; 
}RGB;

RGB int_to_rgb(uint32_t pixel ,  bool  isBigEndian) {
	RGB val; 
	if (isBigEndian) {
		val.r = pixel >> 24 & 0xFF; 
		val.g = pixel >> 16 & 0xFF; 
		val.b = pixel >> 8 & 0xFF; 
		val.a = pixel & 0xFF; 
	}
	else {
		val.a = pixel >> 24 & 0xFF; 
		val.b = pixel >> 16 & 0xFF; 
		val.g = pixel >> 8 & 0xFF; 
		val.r = pixel & 0xFF; 
	}
	return val; 
}


RGB int_to_rgb(uint16_t val, bool isBigEndian) {
	RGB rgb; 
	if (isBigEndian) {
		rgb.r = val >> 12 & 0XF;
		rgb.g = val >> 8 & 0XF;
		rgb.b = val >> 4 & 0XF;
		rgb.a = val & 0XF;


	}
	else {
		rgb.a = val >> 12 & 0XF;
		rgb.b = val >> 8 & 0XF;
		rgb.g = val >> 4 & 0XF;
		rgb.r = val & 0XF;

	}
	return rgb; 
}



RGB int_to_rgb(uint8_t val,  bool isBigEndian) {
	RGB rgb;
	if (isBigEndian) {
		rgb.r = val >> 5 & 0X7;
		rgb.g = val >> 2 & 0X7;
		rgb.b = val & 0X3;
		rgb.a = 0 ;


	}
	else {
		rgb.a = 0;
		rgb.b = val >> 6 & 0X3;
		rgb.g = val >> 3 & 0X7;
		rgb.r = val & 0X7;

	}
	return rgb; 
}


void* rgb_to_int(RGB val,  bool isBigEndian , int bpp) {
	
	if (bpp == 4)
	{
		uint32_t* value = new uint32_t;
		*value = (isBigEndian) ? val.a | (val.b << 8) | (val.g << 16) | (val.r << 24) : val.r | (val.g << 8) | (val.b << 16) | (val.a << 24);
		return value;

	}
else if(bpp == 3 )
	{
		uint32_t* value = new uint32_t;
		*value = (isBigEndian) ?  (val.b ) | (val.g << 8) | (val.r << 16) : val.r | (val.g << 8) | (val.b << 16) ;
		return value;
		
	}
else if(bpp == 2)
	{
		uint16_t* value = new uint16_t;
		*value = (isBigEndian) ? val.a | (val.b << 4) | (val.g << 8) | (val.r << 12) : val.r | (val.g << 4) | (val.b << 8) | (val.a << 12);
		return value;
	}
else if (bpp == 1)
	{
		uint8_t* value = new uint8_t;
		*value = (isBigEndian) ? (val.b) | (val.g << 2) | (val.r << 5) : val.r | (val.g << 2) | (val.b << 5);
		return value;
	}

else return nullptr; 
	
			 	
	

}


__device__
void initialize_2D_array(uint32_t *array, int size_w, int size_h) {
	int i = blockIdx.x;
	int j = threadIdx.x;

	array[i*size_w + j] = 0;
}


template<typename T>
__global__
void GPU_compute_greyscale_luminance(T *array, T *new_array ,int size_w, int size_h,const bool isbigEndian , const int bpp) {
	int i = blockIdx.x; 
	int j = threadIdx.x; 
	RGB rgb = int_to_rgb(array[i], isbigEndian);
	uint8_t value = (int)floor(rgb.r*0.3 + rgb.b*0.11 + rgb.g*0.59);
	rgb.r = value; 
	rgb.g = value; 
	rgb.b = value; 
	rgb.a = 0; 
	void* val = rgb_to_int(rgb, isbigEndian, bpp); 
	new_array[i*size_w + j] =*(static_cast<T*>(val)); 
	delete val; 
}


__global__
void GPU_compute_hmap(uint32_t *array, int size_w, int size_h) {
	initialize_2D_array(array, size_w, size_h); 
	

}



//TODO initialize multi types uint8_t etc ... template 
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


 template <typename T>
 uint32_t* GPU_compute_greyscale(T* image, int width, int height , const int bpp , const bool bigEndian , int pitch ) {

	 T* D_greyscale , *H_greyscale; 
	 T *D_image,*H_image; 
	 H_greyscale = new T[width*height]; 
	 cudaMalloc((void**)&D_greyscale, width*height * sizeof(T)); 
	 cudaMalloc((void**)&D_image, width*height * sizeof(T)); 
	 cudaMemcpy(D_greyscale, H_greyscale, width*height * sizeof(T), cudaMemcpyHostToDevice);
	 cudaMemcpy(D_image, image, width*height * sizeof(T), cudaMemcpyHostToDevice); 
	 int threads = 512; 
	 int blocks_per_grid = floor((width*height + threads - 1) / threads) + 1;
		 GPU_compute_greyscale_luminance << <blocks_per_grid, threads >> > (D_image, D_greyscale , width , height , bigEndian);
	 cudaMemcpy(H_greyscale, D_greyscale, width*height * sizeof(T), cudaMemcpyDeviceToHost);
	 cudaFree(D_new_array); 
	 cudaFree(D_array); 
	 return H_greyscale; 
 }