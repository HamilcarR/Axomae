#include "../includes/ImageManager.h"
#include <iostream>
#include <assert.h>
#include <thread>
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





void ImageManager::set_greyscale_average(SDL_Surface* image,uint8_t factor){
	assert(factor>0);
	assert(image!=nullptr);	
	for(int i = 0 ; i < image->w;i++){
		for(int j = 0 ; j < image->h ; j++){
			RGB rgb = get_pixel_color(image,i,j);
			rgb.red =(rgb.red+rgb.blue+rgb.green)/factor;
			rgb.green =rgb.red;
			rgb.blue = rgb.red;
			uint32_t gray = rgb.rgb_to_int();			
			set_pixel_color(image,i,j,gray);	

		}

	}


}




void ImageManager::set_greyscale_luminance(SDL_Surface* image){
	assert(image!=nullptr);	
	for(int i = 0 ; i < image->w;i++){
		for(int j = 0 ; j < image->h ; j++){
			RGB rgb = get_pixel_color(image,i,j);
			rgb.red =rgb.red*0.3+rgb.blue*0.11+rgb.green*0.59;
			rgb.green =rgb.red;
			rgb.blue = rgb.red;
			uint32_t gray = rgb.rgb_to_int();			
			set_pixel_color(image,i,j,gray);	

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





uint32_t RGB::rgb_to_int(){
	uint32_t image=0 ;
	if(SDL_BYTEORDER == SDL_BIG_ENDIAN){
		image = alpha | (blue << 8) | (green << 16) | (red << 24);	
	}

	else{
	
		image = red | (green << 8) | (blue << 16) | (alpha << 24);	

	}
return image ; 

}










/**************************************************************************************************************/
static auto calculate_kernel_pixel(RGB **data,int kernel[3][3],int i,int j,uint8_t flag){
	if(flag == MAPTOMIX_RED){
		return  data[i-1][j-1].red*kernel[0][0]+data[i][j-1].red*kernel[0][1]+data[i+1][j-1].red*kernel[0][2]+
					 data[i-1][j].red*kernel[1][0] + data[i][j].red*kernel[1][1]+data[i+1][j].red*kernel[1][2]+
					 data[i-1][j+1].red*kernel[2][0]+ data[i][j+1].red*kernel[2][1]+data[i+1][j+1].red*kernel[2][2];			
	}
	else if (flag==MAPTOMIX_BLUE){
		return  data[i-1][j-1].blue*kernel[0][0]+data[i][j-1].blue*kernel[0][1]+data[i+1][j-1].blue*kernel[0][2]+
					 data[i-1][j].blue*kernel[1][0] + data[i][j].blue*kernel[1][1]+data[i+1][j].blue*kernel[1][2]+
					 data[i-1][j+1].blue*kernel[2][0]+ data[i][j+1].blue*kernel[2][1]+data[i+1][j+1].blue*kernel[2][2];	
	}
	else {
		return  data[i-1][j-1].green*kernel[0][0]+data[i][j-1].green*kernel[0][1]+data[i+1][j-1].green*kernel[0][2]+
					 data[i-1][j].green*kernel[1][0] + data[i][j].green*kernel[1][1]+data[i+1][j].green*kernel[1][2]+
					 data[i-1][j+1].green*kernel[2][0]+ data[i][j+1].green*kernel[2][1]+data[i+1][j+1].green*kernel[2][2];	
	}

}



void ImageManager::calculate_edge(SDL_Surface* surface,uint8_t flag,uint8_t border){
	//TODO : use multi threading for initialization and greyscale computing
	/*to avoid concurrent access on image*/
	int w = surface->w;
	int h = surface->h; 
	//thread this :
	set_greyscale_luminance(surface); 


	RGB **data = new RGB*[w];
        for(int i = 0 ; i < w ; i++)
		data[i] = new RGB[h];
	
		
	int max_red = 0,max_blue = 0 ,max_green = 0 ; 
	int min_red = 0,min_blue = 0 , min_green = 0  ; 
	for(int i = 0 ; i < w ; i++){
		for(int j = 0 ; j < h ; j++){

			RGB rgb = get_pixel_color(surface,i,j); 
			max_red = (rgb.red>=max_red) ? rgb.red : max_red ;
			max_green = (rgb.green>=max_green) ? rgb.green : max_green ;
			max_blue = (rgb.blue>=max_blue) ? rgb.blue : max_blue ;
			min_red = (rgb.red<min_red) ? rgb.red : min_red ;
			min_green = (rgb.green<min_green) ? rgb.green : min_green ;
			min_blue = (rgb.blue<min_blue) ? rgb.blue : min_blue ;




			data[i][j].red = rgb.red;
		        data[i][j].blue = rgb.blue;
			data[i][j].green = rgb.green;	
		}

	}
	 thread t1 ; 	
	 int arr_h[3][3];
	 int arr_v[3][3] ;
	for(int i = 0 ;i < 3 ; i ++){
	 for(int j = 0 ; j < 3 ; j++){      
		       arr_v[i][j]=(flag == MAPTOMIX_USE_SOBEL) ? sobel_mask_vertical[i][j] : prewitt_mask_vertical[i][j] ;
		       std::cout<<" val : " << std::to_string(arr_v[i][j]) << "\n";			
		       arr_h[i][j]=(flag == MAPTOMIX_USE_SOBEL) ? sobel_mask_horizontal[i][j] : prewitt_mask_horizontal[i][j] ;
	}
	}
	for(int i = 1 ; i < w-1 ; i++){
		for(int j = 1 ; j < h-1 ; j++){

			if(border== MAPTOMIX_REPEAT){
				int setpix_h_red = 0,setpix_v_red = 0 ,setpix_h_green = 0 , setpix_v_green = 0 , setpix_v_blue = 0 , setpix_h_blue= 0 ; 
			


				setpix_h_red = calculate_kernel_pixel(data,arr_h,i,j,MAPTOMIX_RED);
				setpix_v_red = calculate_kernel_pixel(data,arr_v,i,j,MAPTOMIX_RED);	
				setpix_h_green = calculate_kernel_pixel(data,arr_h,i,j,MAPTOMIX_GREEN);
				setpix_v_green = calculate_kernel_pixel(data,arr_v,i,j,MAPTOMIX_GREEN);				
				setpix_h_blue = calculate_kernel_pixel(data,arr_h,i,j,MAPTOMIX_BLUE);			
				setpix_v_blue =	calculate_kernel_pixel(data,arr_v,i,j,MAPTOMIX_BLUE);
			

				
				
				setpix_v_red=normalize(max_red,min_red,setpix_v_red);
				setpix_h_red=normalize(max_red,min_red,setpix_h_red);
				setpix_v_green=normalize(max_green,min_green,setpix_v_green);
				setpix_h_green=normalize(max_green,min_green,setpix_h_green);
				setpix_v_blue=normalize(max_blue,min_blue,setpix_v_blue);
				setpix_h_blue=normalize(max_blue,min_blue,setpix_h_blue);


				int r=magnitude(setpix_v_red,setpix_h_red); 
				int g=magnitude(setpix_v_green,setpix_h_green);
				int b=magnitude(setpix_v_blue,setpix_h_blue);

				
				RGB rgb = RGB (r,g,b,0);
			    //    rgb.to_string();

				
				set_pixel_color(surface,i,j,rgb.rgb_to_int()); 
				

			}
		}

	}
	

	for(int i = 0 ; i < w;i++)
		delete [] data[i];

	delete [] data; 

}














/**************************************************************************************************************/
}
