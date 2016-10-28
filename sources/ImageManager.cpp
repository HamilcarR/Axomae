#include "../includes/ImageManager.h"
#include <iostream>



namespace maptomix{

	using namespace std;

	ImageManager::ImageManager(){

	}

	ImageManager::~ImageManager(){


	}

/**************************************************************************************************************/


	uint32_t ImageManager::get_pixel_color(SDL_Surface* surface , int x , int y ){

		return ( (uint32_t*) surface->pixels)[y*surface->w+x];
	}





/**************************************************************************************************************/



	void ImageManager::print_pixel(uint32_t color){
		uint8_t red = color >> 24 & 0XFF;
		uint8_t green = color >> 16 & 0XFF;
		uint8_t blue = color >> 8 & 0XFF;
		uint8_t alpha = color & 0XFF;

			cout<<"red : " <<to_string( red )<<"\n"<<"green : "<<to_string(green)<<"\n" <<"blue : \n"<<to_string(blue)<<"\n"<<"alpha : "<<to_string(alpha)<<"\n" <<endl;
	


	}	


/**************************************************************************************************************/

	

	void ImageManager::display_info_surface(SDL_Surface* image){
	
		cout << "Bytes per pixel : " << to_string(image->format->BytesPerPixel) <<endl;
		cout << "Padding on X : " << to_string(image->format->padding[0]) << endl;
		cout << "Padding on Y : " << to_string(image->format->padding[1]) << endl;
}






/**************************************************************************************************************/











/**************************************************************************************************************/
}
