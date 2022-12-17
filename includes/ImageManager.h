#ifndef IMAGEMANAGER_H
#define IMAGEMANAGER_H
#include <SDL2/SDL.h>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <assert.h>
#include "constants.h" 
#include "../kernels/Kernel.cuh" 

namespace axomae{
			
			
	class RGB{
		public:
			RGB();
			RGB(int r , int g , int b , int a);
			RGB(int r, int g , int b);
			~RGB();
			static RGB int_to_rgb(uint32_t value);
			static RGB int_to_rgb(uint8_t value);
			static RGB int_to_rgb(uint16_t value);
		        double intensity();
			void invert_color();			
			RGB operator+=(int arg);
			RGB operator+(RGB arg);
			RGB operator/(int arg);
			uint32_t rgb_to_int();
			void to_string();
			int red;
			int green;
			int blue;
			int alpha;
	};



 
	class ImageManager{
		
		public:
			static max_colors* get_colors_max_variation(SDL_Surface* image);

			static void compute_edge(SDL_Surface* greyscale_surface,uint8_t flag,uint8_t border_behaviour );
			static RGB get_pixel_color(SDL_Surface* surface,int x,int y);
			static void print_pixel(uint32_t color);	
			static void display_info_surface(SDL_Surface* image);
			static void set_pixel_color(SDL_Surface* image,int x,int y , uint32_t color);
			static void set_pixel_color(SDL_Surface* image,RGB **pixel_array,int w,int h);
			static void set_greyscale_average(SDL_Surface* image,uint8_t factor);	
			static void set_greyscale_luminance(SDL_Surface* image);
			static void set_contrast(SDL_Surface* image,int level);
			static void set_contrast(SDL_Surface* image);
			static void set_contrast_sigmoid(SDL_Surface *image,int treshold);
			static void compute_normal_map(SDL_Surface* surface , double strength , float attenuation); 		
			static void compute_dudv(SDL_Surface* surface,double factor);
			static void USE_GPU_COMPUTING() { gpu = true; }
			static void USE_CPU_COMPUTING() { gpu = false;  }
			static bool USING_GPU() { return gpu;  }
			static SDL_Surface* copy_surface(SDL_Surface* src); 
			static SDL_Surface* project_uv_normals(Object3D object,  int width , int height , bool tangent_space);		    
		private:


			ImageManager();
			~ImageManager();

			static bool gpu;

	};
	
	


}

























#endif
