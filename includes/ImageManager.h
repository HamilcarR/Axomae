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
			constexpr uint8_t MAPTOMIX_USE_SCHARR = 0X01;
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
			

			static constexpr int scharr_vertical[KERNEL_SIZE][KERNEL_SIZE]={
				{3 , 10 , 3},
				{0 , 0  , 0},
				{-3, -10 ,-3}



			};


			static constexpr int scharr_horizontal[KERNEL_SIZE][KERNEL_SIZE]={
				{3 , 0 , -3},
				{10 , 0 , -10},
				{3 , 0 , -3}


			};



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
	
/*			static auto normalize_0_1 = [](RGB rgb){
					RGB ret ; 
					ret.red = rgb.red/255;
					ret.green = rgb.red/255;
					ret.blue = rgb.red/255;
					ret.alpha = rgb.alpha; 
					return ret;

			};	
			
			static auto normalize_0_255 =[](RGB rgb){
					RGB ret;
				       	ret.red=rgb.red*255;
					ret.green=rgb.green*255;	
					ret.blue = rgb.blue*255;
					ret.alpha = rgb.alpha;
					return ret;
			};

*/

	typedef struct max_colors{

		int max_rgb[3];
		int min_rgb[3];	
	
	};			
			
	class RGB{

		public:
			RGB();
			RGB(int r , int g , int b , int a);
			~RGB();
			static RGB int_to_rgb(uint32_t value);
			static RGB int_to_rgb(uint8_t value);
			static RGB int_to_rgb(uint16_t value);
			void invert_color();
			
			template<typename T>
			static RGB invert_color(T &color);
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
			static void calculate_edge(SDL_Surface* surface,uint8_t flag,uint8_t border_behaviour);
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
		private:

			ImageManager();
			~ImageManager();


	};






}

























#endif
