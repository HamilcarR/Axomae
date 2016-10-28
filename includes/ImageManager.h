#ifndef IMAGEMANAGER_H
#define IMAGEMANAGER_H

#include <SDL2/SDL.h>





namespace maptomix{


	class ImageManager{

		public:
			static uint32_t get_pixel_color(SDL_Surface* surface,int x,int y);
			static void print_pixel(uint32_t color);	
			static void display_info_surface(SDL_Surface* image);


		private:

			ImageManager();
			~ImageManager();





	};














}

























#endif
