#ifndef IMAGEMANAGER_H
#define IMAGEMANAGER_H

#include <SDL2/SDL.h>





namespace maptomix{




	class RGB{

		public:
			RGB();
			RGB(uint8_t r , uint8_t g , uint8_t b , uint8_t a);
			~RGB();
			static RGB int_to_rgb(uint32_t value);
			static RGB int_to_rgb(uint8_t value);
			static RGB int_to_rgb(uint16_t value);
			auto rgb_to_int();
			void to_string();
			uint8_t red;
			uint8_t green;
			uint8_t blue;
			uint8_t alpha;
	};





	class ImageManager{

		public:
			static RGB get_pixel_color(SDL_Surface* surface,int x,int y);
			static void print_pixel(uint32_t color);	
			static void display_info_surface(SDL_Surface* image);
			static void set_pixel_color(SDL_Surface* image,int x,int y , uint32_t color);
			static void set_greyscale(SDL_Surface* image);	
			
		private:

			ImageManager();
			~ImageManager();


	};






}

























#endif
