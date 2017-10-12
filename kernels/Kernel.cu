#include "Kernel.cuh"
#include <cmath>

typedef struct RGB {
	uint8_t r;
	uint8_t g; 
	uint8_t b; 
	uint8_t a; 
}RGB;




__device__
RGB int_to_rgb(uint32_t pixel ,  bool  isBigEndian , int bpp) {
	RGB val; 
	if (bpp == 4) {
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
	else if (bpp == 3) {
		if (isBigEndian) {
			val.r = pixel >> 16 & 0XFF;
			val.g = pixel >> 8 & 0XFF;
			val.b = pixel & 0XFF;


		}
		else {
			val.b = pixel >> 16 & 0XFF;
			val.g = pixel >> 8 & 0XFF;
			val.r = pixel & 0XFF;

		}
		return val;
	}
	else {
		return { 0 , 0 , 0 , 0 };
	}
}



__device__
RGB int_to_rgb(uint16_t val, bool isBigEndian) {
	RGB rgb;
	if (isBigEndian) {
		rgb.r = val >> 12 & 0xF;
		rgb.g = val >> 8 & 0XF;
		rgb.b = val >> 4 & 0XF;
		rgb.a = val & 0XF;



	}
	else {
		rgb.a = val >> 12 & 0xF;
		rgb.b = val >> 8 & 0XF;
		rgb.g = val >> 4 & 0XF;
		rgb.r = val & 0XF;

	}
	return rgb;
}

__device__
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

__device__
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







 

 __global__
	 void GPU_compute_greyscale_luminance(void *array, int size_w, int size_h, const bool isbigEndian, const int bpp , int pitch) {
	 int j = blockIdx.x;
	 int i = threadIdx.x;
	 RGB rgb = { 0 , 0 , 0 , 0 };
	 if (i < size_w && j < size_h) {
		 uint8_t* pixel_value = (uint8_t*) (array)+i*bpp + j*pitch;
		 if (bpp == 4) {
			 rgb = int_to_rgb(*pixel_value, isbigEndian , 4); 
			 rgb.r = (rgb.r + rgb.b + rgb.g) / 3;
			 rgb.g = rgb.r;
			 rgb.b = rgb.r;
			 uint32_t* toInt =(uint32_t*) rgb_to_int(rgb, isbigEndian, bpp); 
			 *(uint32_t*)(pixel_value) = *toInt; 
			 delete toInt; 

		 }
		 else if (bpp == 3) {
			 rgb = int_to_rgb(*(uint32_t*)pixel_value, isbigEndian , 3);
			 rgb.r = (rgb.r + rgb.b + rgb.g) / 3;
			 rgb.g = rgb.r;
			 rgb.b = rgb.r;
			 uint32_t* toInt = (uint32_t*)rgb_to_int(rgb, isbigEndian, bpp);
			 if (!isbigEndian) {
				 ((uint8_t*)pixel_value)[0] = *toInt >> 16 & 0xFF;
				 ((uint8_t*)pixel_value)[1] = *toInt >> 8 & 0xFF; 
				 ((uint8_t*)pixel_value)[2] = *toInt & 0xFF; 
				  
			 }
			 else {
				 ((uint8_t*)pixel_value)[0] = *toInt  & 0xFF;
				 ((uint8_t*)pixel_value)[1] = *toInt >> 8 & 0xFF;
				 ((uint8_t*)pixel_value)[2] = *toInt >> 16 & 0xFF;
			 }

			 delete toInt;
		 }
		 else if (bpp == 2) {
			 rgb = int_to_rgb(*(uint16_t*)pixel_value, isbigEndian);
			 rgb.r = (rgb.r + rgb.b + rgb.g) / 3;
			 rgb.g = rgb.r;
			 rgb.b = rgb.r;
			 uint32_t* toInt = (uint32_t*)rgb_to_int(rgb, isbigEndian, bpp);
			 *((uint16_t*)pixel_value) = *toInt; 
			 delete toInt; 
		 }
		 else if (bpp == 1) {
			 rgb = int_to_rgb(*pixel_value, isbigEndian);
			 rgb.r = (rgb.r + rgb.b + rgb.g) / 3;
			 rgb.g = rgb.r;
			 rgb.b = rgb.r;
			 uint32_t* toInt = (uint32_t*)rgb_to_int(rgb, isbigEndian, bpp);
			 *((uint8_t*)pixel_value) = *toInt;
			 delete toInt;
		 }
		

	 }
	 
	 

 }









 void GPU_compute_greyscale(SDL_Surface* image,  const bool bigEndian) {
	 int width = image->w; 
	 int height = image->h; 
	 int pitch = image->pitch; 
	 int bpp = image->format->BytesPerPixel; 
	 
	 uint8_t* D_image; 
	 size_t size = (width - 1)*bpp + (height - 1)*pitch; 
	 cudaMalloc((void**)&D_image, size); 
	 cudaMemcpy(D_image, image->pixels, size, cudaMemcpyHostToDevice);

	 int threads = 512;
	 int blocks_per_grid = floor((width*height + threads - 1) / threads) + 1;
	 GPU_compute_greyscale_luminance << <blocks_per_grid, threads >> > (D_image, width, height, bigEndian, bpp, pitch);
	 SDL_LockSurface(image); 
	 cudaMemcpy(image->pixels, D_image, size, cudaMemcpyDeviceToHost);
	 SDL_UnlockSurface(image); 
	 cudaFree(D_image); 
 }




 __global__
	 void GPU_compute_hmap(uint32_t *array, int size_w, int size_h) {
	 initialize_2D_array(array, size_w, size_h);


 }
