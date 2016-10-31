#ifndef IMAGEMANAGER_H
#define IMAGEMANAGER_H

#include <SDL2/SDL.h>
#include <algorithm>
#include <iostream>
#include <cmath>


namespace maptomix{

			/*we define some constants here...flags,sobel-prewitt operators,kernels etc...*/
			constexpr uint8_t MAPTOMIX_USE_SOBEL = 0XFF;
			constexpr uint8_t MAPTOMIX_USE_PREWITT = 0X00;
			constexpr uint8_t MAPTOMIX_CLAMP = 0XFF;
			constexpr uint8_t MAPTOMIX_REPEAT = 0X00;
			constexpr uint8_t MAPTOMIX_MIRROR = 0X01;
			constexpr uint8_t MAPTOMIX_GHOST = 0X02;
			constexpr uint8_t MAPTOMIX_RED = 0X00;
			constexpr uint8_t MAPTOMIX_GREEN = 0X01; 
			constexpr uint8_t MAPTOMIX_BLUE = 0X02;
			constexpr uint8_t KERNEL_SIZE = 3 ; 
			 //Operators
			static constexpr int SOBEL= 2 ; 
			static constexpr int PREWITT = 1 ; 

			//convolution kernels

			static constexpr int sobel_mask_vertical[][3] = { 	
				{-1,-SOBEL,-1},
				{0,  0  ,0},
				{1, SOBEL , 1}		

			};	
			static constexpr int sobel_mask_horizontal[][3]={
				{1,0,-1},
				{SOBEL,0,-SOBEL},
				{1,0,-1}
				

			};
			
			static constexpr int prewitt_mask_vertical[][3] = {

				{1,PREWITT,1},
				{0,  0  ,0},
				{-1, -PREWITT , -1}		

			};	
			static constexpr int prewitt_mask_horizontal[][3]={
				{-1,0,1},
				{-PREWITT,0,PREWITT},
				{-1,0,1}
				

			};
			static auto normalize = [](int maxx,int minn,int pixel){
					return ( (pixel-minn)*255 / (maxx-minn) + 0 );
				};	
	
			static auto magnitude = [](int x,int y){return sqrt(x*x+y*y);};
	class RGB{

		public:
			RGB();
			RGB(uint8_t r , uint8_t g , uint8_t b , uint8_t a);
			~RGB();
			static RGB int_to_rgb(uint32_t value);
			static RGB int_to_rgb(uint8_t value);
			static RGB int_to_rgb(uint16_t value);
			
			uint32_t rgb_to_int();
			void to_string();
			uint8_t red;
			uint8_t green;
			uint8_t blue;
			uint8_t alpha;
	};





	class ImageManager{
		
		public:
					
			static void calculate_edge(SDL_Surface* surface,uint8_t flag,uint8_t border_behaviour);
			static RGB get_pixel_color(SDL_Surface* surface,int x,int y);
			static void print_pixel(uint32_t color);	
			static void display_info_surface(SDL_Surface* image);
			static void set_pixel_color(SDL_Surface* image,int x,int y , uint32_t color);
			static void set_greyscale_average(SDL_Surface* image,uint8_t factor);	
			static void set_greyscale_luminance(SDL_Surface* image);
		private:

			ImageManager();
			~ImageManager();


	};






}

























#endif
