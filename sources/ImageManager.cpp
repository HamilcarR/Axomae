#include "../includes/ImageManager.h"
#include <iostream>


namespace maptomix{

	using namespace std;



	RGB::RGB():red(0),green(0),blue(0),alpha(0){
		
	}

	RGB::RGB(uint8_t r,uint8_t g,uint8_t b,uint8_t a):red(r),green(g),blue(b),alpha(a){

	}

	RGB::~RGB(){


	}




	ImageManager::ImageManager(){

	}

	ImageManager::~ImageManager(){


	}

/**************************************************************************************************************/


	RGB ImageManager::get_pixel_color(SDL_Surface* surface , int x , int y ){

		int bpp = surface->format->BytesPerPixel;
		
		uint8_t *color = (uint8_t*) (surface->pixels) + x*bpp + y*surface->pitch;
		RGB rgb=RGB();

		switch(bpp){
			case 1 :
			;
			rgb.red = *color >> 5 & 0x7;
		        rgb.green = *color >> 2 & 0x7;
			rgb.blue = *color & 0x3 ;	
			break;


			case 2:
			;{
			uint16_t colo16bits  =* (uint16_t*) color ; 
			if(SDL_BYTEORDER == SDL_BIG_ENDIAN){

				rgb.red =  colo16bits >> 12 & 0xF;
				rgb.green = colo16bits >> 8 & 0XF;
				rgb.blue = colo16bits >> 4 & 0XF;
				rgb.alpha = colo16bits & 0XF; 
			}
			else{
	
				rgb.alpha =  colo16bits >> 12 & 0xF;
				rgb.blue = colo16bits >> 8 & 0XF;
				rgb.green = colo16bits >> 4 & 0XF;
				rgb.red = colo16bits & 0XF; 				

			}
			}
			break;


			case 3:
			;{
			uint32_t colo24bits = *(uint32_t*) color ; 
			if(SDL_BYTEORDER == SDL_BIG_ENDIAN){
				rgb.red = colo24bits >> 16 & 0XFF;
				rgb.green = colo24bits >> 8 & 0XFF;
				rgb.blue = colo24bits & 0XFF ; 

			}

			else{

				rgb.blue = colo24bits >> 16 & 0XFF;
				rgb.green = colo24bits >> 8 & 0XFF;
				rgb.red = colo24bits & 0XFF ; 

			}	

			}
			break;


			case 4:
			;{
			uint32_t colo32bits = *(uint32_t*) color ; 
			if(SDL_BYTEORDER == SDL_BIG_ENDIAN){
				
				rgb.red = colo32bits >> 24 & 0XFF;
				rgb.green = colo32bits >> 16 & 0XFF;
				rgb.blue = colo32bits >> 8 & 0XFF;
				rgb.alpha = colo32bits & 0XFF ; 

			}

			else{

				rgb.alpha = colo32bits >> 24 & 0XFF;
				rgb.blue = colo32bits >> 16 & 0XFF;
				rgb.green = colo32bits >> 8 & 0XFF;
				rgb.red = colo32bits & 0XFF ; 


			}
			}

			break;


		}
		return rgb;
	
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


	void ImageManager::set_pixel_color(SDL_Surface* surface ,int x,int y, uint32_t color){	
		int bpp = surface->format->BytesPerPixel;
	
		SDL_LockSurface(surface);

			 Uint8* pix= &((Uint8*)surface->pixels) [ x*bpp + y*surface->pitch ];
			
			if(bpp==1)
			{
				Uint8* pixel =(Uint8*) pix;
				*pixel = color ; 


			}
			else if(bpp==2){
				*((Uint16*) pix) = color ;

			}

			else if(bpp==3){
				if(SDL_BYTEORDER == SDL_BIG_ENDIAN){
					((Uint8*)pix)[0] = color >> 16 & 0XFF;
					((Uint8*)pix)[1] = color >> 8 & 0XFF;
				        ((Uint8*)pix)[2] = color & 0XFF;

				}
				else{
				
					((Uint8*)pix)[0] = color & 0XFF;
					((Uint8*)pix)[1] = color >> 8 & 0XFF;
				        ((Uint8*)pix)[2] = color >>16 & 0XFF;

				}
			
			}

			else{
				*((Uint32*) pix) = color ; 
			}

		SDL_UnlockSurface(surface);

	}








/**************************************************************************************************************/


void RGB::to_string(){
	cout << "RED : " << std::to_string(red)<<"\n";
	cout << "GREEN : " << std::to_string(green)<<"\n";
	cout << "BLUE : " << std::to_string(blue)<<"\n";
	cout << "ALPHA : " << std::to_string(alpha)<<"\n";

}





















/**************************************************************************************************************/





void ImageManager::set_greyscale(SDL_Surface* image){
	
	for(int i = 0 ; i < image->w;i++){
		for(int j = 0 ; j < image->h ; j++){
			RGB rgb = get_pixel_color(image,i,j);
			

		}

	}









}




















/**************************************************************************************************************/



RGB RGB::int_to_rgb(uint32_t val){
	RGB rgb = RGB(); 

	if(SDL_BYTEORDER == SDL_BIG_ENDIAN){
		rgb.red = val >> 24 & 0XFF ; 
		rgb.green = val >> 16 & 0XFF ; 
		rgb.blue = val >> 8 & 0XFF ; 
		rgb.alpha = val & 0XFF;


	}
	else{
		rgb.alpha = val >> 24 & 0XFF ; 
		rgb.blue = val >> 16 & 0XFF ; 
		rgb.green = val >> 8 & 0XFF ; 
		rgb.red = val & 0XFF;

	}
return rgb; 

}





RGB RGB::int_to_rgb(uint16_t val){
	RGB rgb = RGB(); 

	if(SDL_BYTEORDER == SDL_BIG_ENDIAN){
		rgb.red = val >> 12 & 0XF ; 
		rgb.green = val >> 8 & 0XF ; 
		rgb.blue = val >> 4 & 0XF ; 
		rgb.alpha = val & 0XF;


	}
	else{
		rgb.alpha = val >> 12 & 0XF ; 
		rgb.blue = val >> 8 & 0XF ; 
		rgb.green = val >> 4 & 0XF ; 
		rgb.red = val & 0XF;

	}
return rgb; 

}





auto RGB::rgb_to_int(){
	uint32_t image=0 ;
	if(SDL_BYTEORDER == SDL_BIG_ENDIAN){
		image = alpha | (blue << 8) | (green << 16) | (red << 24);	
		cout << "big endian " <<"\n";
	}

	else{
	
		image = red | (green << 8) | (blue << 16) | (red << 24);	

	}
return image ; 

}










/**************************************************************************************************************/
/**************************************************************************************************************/
}
