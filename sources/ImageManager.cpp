#include "../includes/ImageManager.h"
#include "../kernels/Kernel.cuh"
#include <iostream>
#include <assert.h>
#include <math.h>
#include <thread>
#include <cstdlib>
#include <climits>
#include <string>
#include <ctime>

namespace axomae{
	


	using namespace std;
	bool ImageManager::gpu = false;

	 static bool CHECK_IF_CUDA_AVAILABLE() {
		if  (CUDART_VERSION != 9000 ) 
			return false;

	    if (!ImageManager::USING_GPU()) 
				return false; 
		else 
				return true;

		
	}


	template<typename T>
	static void replace_image(SDL_Surface* surface, T* image, unsigned int size, int bpp);



	RGB::RGB():red(0),green(0),blue(0),alpha(0){
		
	}

	RGB::RGB(int r , int g , int b):red(r),green(g),blue(b),alpha(0){
	}

	RGB::RGB(int r , int g , int b , int a):red(r),green(g),blue(b),alpha(a){

	}

	RGB::~RGB(){


	}




	ImageManager::ImageManager(){

	}

	ImageManager::~ImageManager(){


	}

/**************************************************************************************************************/

	int truncate(int n){
		if(n<=0)
	  		return 0;
		else if (n>=255)
	 		 return 255;
		else
	  		return n ; 

}


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
				; {
					uint16_t colo16bits = *(uint16_t*)color;
					if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {

						rgb.red = colo16bits >> 12 & 0xF;
						rgb.green = colo16bits >> 8 & 0XF;
						rgb.blue = colo16bits >> 4 & 0XF;
						rgb.alpha = colo16bits & 0XF;
					}
					else {

						rgb.alpha = colo16bits >> 12 & 0xF;
						rgb.blue = colo16bits >> 8 & 0XF;
						rgb.green = colo16bits >> 4 & 0XF;
						rgb.red = colo16bits & 0XF;

					}
				}
			break;
			case 3:
			{
				uint32_t colo24bits = *(uint32_t*)color;
				if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
					rgb.red = colo24bits >> 16 & 0XFF;
					rgb.green = colo24bits >> 8 & 0XFF;
					rgb.blue = colo24bits & 0XFF;

				}

				else {

					rgb.blue = colo24bits >> 16 & 0XFF;
					rgb.green = colo24bits >> 8 & 0XFF;
					rgb.red = colo24bits & 0XFF;
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

	void ImageManager::set_pixel_color(SDL_Surface* surface, int x, int y, uint32_t color) {
		int bpp = surface->format->BytesPerPixel;
		SDL_LockSurface(surface);
		Uint8* pix = &((Uint8*)surface->pixels)[x*bpp + y*surface->pitch];
		if (bpp == 1)
		{
			Uint8* pixel = (Uint8*)pix;
			*pixel = color;
		}
		else if (bpp == 2) 
			*((Uint16*)pix) = color;
		

		else if (bpp == 3) {
			if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
				((Uint8*)pix)[0] = color >> 16 & 0XFF;
				((Uint8*)pix)[1] = color >> 8 & 0XFF;
				((Uint8*)pix)[2] = color & 0XFF;
			}
			else {
				((Uint8*)pix)[0] = color & 0XFF;
				((Uint8*)pix)[1] = color >> 8 & 0XFF;
				((Uint8*)pix)[2] = color >> 16 & 0XFF;
			}
		}

		else 
			*((Uint32*)pix) = color;
		SDL_UnlockSurface(surface);

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


	void ImageManager::set_pixel_color(SDL_Surface* surface,RGB **arrayc,int w,int h){
		for(int i = 0 ; i < w ; i++){
			for(int j = 0 ; j < h ; j++){
				set_pixel_color(surface,i,j,arrayc[i][j].rgb_to_int());

			}
			
		}
	
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
	if (CHECK_IF_CUDA_AVAILABLE()) {
		GPU_compute_greyscale(image, SDL_BYTEORDER == SDL_BIG_ENDIAN , false); 
	}
	else {
		for (int i = 0; i < image->w; i++) {
			for (int j = 0; j < image->h; j++) {
				RGB rgb = get_pixel_color(image, i, j);
				rgb.red = (rgb.red + rgb.blue + rgb.green) / factor;
				rgb.green = rgb.red;
				rgb.blue = rgb.red;
				uint32_t gray = rgb.rgb_to_int();
				set_pixel_color(image, i, j, gray);

			}

		}
	}


}


static void replace_image(SDL_Surface* surface, uint8_t* image) {
	//TODO LOCK SURFACE ? 
	int bpp = surface->format->BytesPerPixel; 
	int size = surface->w * surface->h;

	SDL_LockSurface(surface);
	 if (bpp == 1) {
		 for ( int i = 0; i < surface->w; i++)
			 for ( int j = 0; j < surface->h; j++)
				 ((Uint8*)surface->pixels)[i*bpp + j*surface->pitch] = image[i*bpp + j*surface->pitch];

		delete[] static_cast<uint8_t*> (image);

	}
	else {
		std::cout << "error reading image ... BPP is : " << std::to_string(bpp) << " Bytes per pixel\n";
	}
	SDL_UnlockSurface(surface);

}


static void replace_image(SDL_Surface* surface, uint16_t* image) {
	//TODO LOCK SURFACE ? 
	int bpp = surface->format->BytesPerPixel;
	int size = surface->w * surface->h;
	SDL_LockSurface(surface);
	 if (bpp == 2) {
		 for ( int i = 0; i < surface->w; i++)
			 for ( int j = 0; j < surface->h; j++)
				 ((Uint16*)surface->pixels)[i*bpp + j*surface->pitch] = image[i*bpp + j*surface->pitch];
		delete[]  static_cast<uint16_t*> (image);

	}
	
	else {
		std::cout << "error reading image ... BPP is : " << std::to_string(bpp) << " Bytes per pixel\n";
	}
	SDL_UnlockSurface(surface);

}


static void replace_image(SDL_Surface* surface, uint32_t* image) {
	//TODO LOCK SURFACE ? 
	int bpp = surface->format->BytesPerPixel; 
	int size = surface->w * surface->h; 
	int pitch = surface->pitch; 
	SDL_LockSurface(surface);
	if (bpp == 4) {
		for ( int i = 0; i < surface->w; i++)
			for ( int j = 0; j < surface->h; j++) 
				((Uint32*)surface->pixels)[i*bpp + j*surface->pitch] = image[i*bpp + j*surface->pitch];
			
		delete[] static_cast<uint32_t*> (image);

	}
	else if (bpp == 3) {
		for ( int i = 0; i < surface->w; i++)
			for ( int j = 0; j < surface->h; j++)
				((Uint32*)surface->pixels)[i*bpp + j*surface->pitch] = image[i*bpp + j*surface->pitch];

	}

	
	else {
		std::cout << "error reading image ... BPP is : " << std::to_string(bpp) << " Bytes per pixel\n";
	}
	SDL_UnlockSurface(surface);

}
void ImageManager::set_greyscale_luminance(SDL_Surface* image){
	bool cuda = CHECK_IF_CUDA_AVAILABLE() ; 
	std::clock_t clock;

	if (cuda) {
		clock = std::clock(); 
		GPU_compute_greyscale(image, SDL_BIG_ENDIAN == SDL_BYTEORDER , true);
		std::cout << (std::clock() - clock) / (double)CLOCKS_PER_SEC << "\n"; 
	}

	
	else {
		assert(image != nullptr);
		clock = std::clock();
		for (int i = 0; i < image->w; i++) 
			for (int j = 0; j < image->h; j++) {
				RGB rgb = get_pixel_color(image, i, j);
				rgb.red = (int)floor(rgb.red*0.3 + rgb.blue*0.11 + rgb.green*0.59);
				rgb.green = rgb.red;
				rgb.blue = rgb.red;
				uint32_t gray = rgb.rgb_to_int();
				set_pixel_color(image, i, j, gray);

			}
		std::cout << (std::clock() - clock) / (double)CLOCKS_PER_SEC << "\n";


		
	}


}







SDL_Surface* ImageManager::copy_surface(SDL_Surface *src) {
	SDL_Surface* res; 
	res = SDL_CreateRGBSurface(src->flags, src->w, src->h, src->format->BitsPerPixel, src->format->Rmask, src->format->Gmask, src->format->Bmask, src->format->Amask); 
	if (res != nullptr) {
		SDL_BlitSurface(src, nullptr, res, nullptr); 
		return res;
	}
	else
		return nullptr; 
}















/**************************************************************************************************************/
const double RGB::intensity(){

const double av= (red+green+blue)/3;

return av/255;


}


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



void ImageManager::compute_edge(SDL_Surface* surface,uint8_t flag,uint8_t border){
	 bool cuda = CHECK_IF_CUDA_AVAILABLE(); 

	if  (cuda) {
		GPU_compute_greyscale(surface, SDL_BYTEORDER == SDL_BIG_ENDIAN , true); 
	}

	else  {
		//TODO : use multi threading for initialization and greyscale computing
		/*to avoid concurrent access on image*/
		int w = surface->w;
		int h = surface->h;
		//thread this :
		set_greyscale_luminance(surface);


		RGB **data = new RGB*[w];
		for (int i = 0; i < w; i++)
			data[i] = new RGB[h];


		int max_red = 0, max_blue = 0, max_green = 0;
		int min_red = 0, min_blue = 0, min_green = 0;
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < h; j++) {

				RGB rgb = get_pixel_color(surface, i, j);
				max_red = (rgb.red >= max_red) ? rgb.red : max_red;
				max_green = (rgb.green >= max_green) ? rgb.green : max_green;
				max_blue = (rgb.blue >= max_blue) ? rgb.blue : max_blue;
				min_red = (rgb.red < min_red) ? rgb.red : min_red;
				min_green = (rgb.green < min_green) ? rgb.green : min_green;
				min_blue = (rgb.blue < min_blue) ? rgb.blue : min_blue;




				data[i][j].red = rgb.red;
				data[i][j].blue = rgb.blue;
				data[i][j].green = rgb.green;
			}

		}

		int arr_h[3][3];
		int arr_v[3][3];
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (flag == MAPTOMIX_USE_SOBEL) {
					arr_v[i][j] = sobel_mask_vertical[i][j];
					arr_h[i][j] = sobel_mask_horizontal[i][j];
				}
				else if (flag == MAPTOMIX_USE_PREWITT) {
					arr_h[i][j] = prewitt_mask_horizontal[i][j];
					arr_v[i][j] = prewitt_mask_vertical[i][j];

				}
				else if (flag == MAPTOMIX_USE_SCHARR) {
					arr_h[i][j] = scharr_horizontal[i][j];
					arr_v[i][j] = scharr_vertical[i][j];

				}

			}
		}
		for (int i = 1; i < w - 1; i++) {
			for (int j = 1; j < h - 1; j++) {

				if (border == MAPTOMIX_REPEAT) {
					int setpix_h_red = 0, setpix_v_red = 0, setpix_h_green = 0, setpix_v_green = 0, setpix_v_blue = 0, setpix_h_blue = 0;



					setpix_h_red = calculate_kernel_pixel(data, arr_h, i, j, MAPTOMIX_RED);
					setpix_v_red = calculate_kernel_pixel(data, arr_v, i, j, MAPTOMIX_RED);
					setpix_h_green = calculate_kernel_pixel(data, arr_h, i, j, MAPTOMIX_GREEN);
					setpix_v_green = calculate_kernel_pixel(data, arr_v, i, j, MAPTOMIX_GREEN);
					setpix_h_blue = calculate_kernel_pixel(data, arr_h, i, j, MAPTOMIX_BLUE);
					setpix_v_blue = calculate_kernel_pixel(data, arr_v, i, j, MAPTOMIX_BLUE);




					setpix_v_red = normalize(max_red, min_red, setpix_v_red);
					setpix_h_red = normalize(max_red, min_red, setpix_h_red);
					setpix_v_green = normalize(max_green, min_green, setpix_v_green);
					setpix_h_green = normalize(max_green, min_green, setpix_h_green);
					setpix_v_blue = normalize(max_blue, min_blue, setpix_v_blue);
					setpix_h_blue = normalize(max_blue, min_blue, setpix_h_blue);


					int r = (int)magnitude(setpix_v_red, setpix_h_red);
					int g = (int)magnitude(setpix_v_green, setpix_h_green);
					int b = (int)magnitude(setpix_v_blue, setpix_h_blue);


					RGB rgb = RGB(r, g, b, 0);


					set_pixel_color(surface, i, j, rgb.rgb_to_int());


				}
			}


		}


		//we invert the color of the image 
		for (int i = 1; i < w - 1; i++) {
			for (int j = 1; j < h - 1; j++) {
				RGB rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8;
				rgb1 = get_pixel_color(surface, i - 1, j - 1);
				rgb2 = get_pixel_color(surface, i, j - 1);
				rgb3 = get_pixel_color(surface, i + 1, j - 1);
				rgb4 = get_pixel_color(surface, i - 1, j);
				rgb5 = get_pixel_color(surface, i + 1, j);
				rgb6 = get_pixel_color(surface, i + 1, j - 1);
				rgb7 = get_pixel_color(surface, i + 1, j);
				rgb8 = get_pixel_color(surface, i + 1, j + 1);



				static auto normalize_rgb = [max_red, max_green, max_blue, min_red, min_green, min_blue](RGB rgb) {
					RGB rgb_ret;
					rgb_ret.red = normalize(max_red, min_red, rgb.red);
					rgb_ret.green = normalize(max_green, min_green, rgb.green);
					rgb_ret.blue = normalize(max_blue, min_blue, rgb.blue);
					return rgb_ret;

				};
				rgb1 = normalize_rgb(rgb1);
				rgb2 = normalize_rgb(rgb2);
				rgb3 = normalize_rgb(rgb3);
				rgb4 = normalize_rgb(rgb4);
				rgb5 = normalize_rgb(rgb5);
				rgb6 = normalize_rgb(rgb6);
				rgb7 = normalize_rgb(rgb7);
				rgb8 = normalize_rgb(rgb8);

				RGB center = (rgb1 + rgb2 + rgb3 + rgb4 + rgb5 + rgb6 + rgb7 + rgb8) / 8;

				center.invert_color();

				data[i][j] = center;

			}
		}

		set_pixel_color(surface, data, w, h);


		for (int i = 0; i < w; i++)
			delete[] data[i];

		delete[] data;
	}

}














/**************************************************************************************************************/



RGB RGB::operator+=(int arg){
	RGB rgb=RGB(red+arg,green+arg,blue+arg,alpha+arg);
	return rgb;
	
}


RGB RGB::operator+(RGB arg){
	
	

	RGB rgb = RGB(red+arg.red,green+arg.green,blue+arg.blue,alpha+arg.alpha);
	return rgb;


}


RGB RGB::operator/(int arg){
	assert(arg>0);
	RGB rgb = RGB(red/arg , green/arg, blue/arg,alpha/arg);
	return rgb;
}






/**************************************************************************************************************/

void RGB::invert_color(){


  red=abs(red-255);
  green=abs(green-255) ;
  blue=abs(blue-255);
  alpha=abs(alpha-255);
}

template<typename T>
RGB RGB::invert_color(T &color){

	
}









/**************************************************************************************************************/


max_colors *ImageManager::get_colors_max_variation(SDL_Surface* image){
	max_colors *max_min = new max_colors ;
        const int INT_MAXX = 0;	
	int max_red = 0 , max_green = 0 , max_blue = 0 , min_red = INT_MAX , min_blue = INT_MAX , min_green = INT_MAXX;
	for(int i = 0 ; i < image->w ; i++){
		for(int j = 0 ; j < image->h ; j++){

			RGB rgb = get_pixel_color(image,i,j); 
			max_red = (rgb.red>=max_red) ? rgb.red : max_red ;
			max_green = (rgb.green>=max_green) ? rgb.green : max_green ;
			max_blue = (rgb.blue>=max_blue) ? rgb.blue : max_blue ;
			min_red = (rgb.red<min_red) ? rgb.red : min_red ;
			min_green = (rgb.green<min_green) ? rgb.green : min_green ;
			min_blue = (rgb.blue<min_blue) ? rgb.blue : min_blue ;
		}
	}

	max_min->max_rgb[0] = max_red;
	max_min->max_rgb[1] = max_green;
	max_min->max_rgb[2] = max_blue ; 

	max_min->min_rgb[0] = min_red;
	max_min->min_rgb[1] = min_green;
	max_min->min_rgb[2] = min_blue; 

	return max_min ; 

}

void ImageManager::set_contrast(SDL_Surface* image,int level){
	double correction_factor = (259*(level+255))/(255*(259-level));
	max_colors *maxmin = get_colors_max_variation(image);
	for(int i = 0 ; i < image->w; i++){
		for(int j = 0 ; j < image->h; j++){
			RGB col = get_pixel_color(image,i,j);
			col.red = (int) floor(truncate(correction_factor * (col.red - 128)+128));
			col.green = (int) floor(truncate(correction_factor * (col.green - 128)+128));
			col.blue = (int) floor(truncate(correction_factor * (col.blue - 128)+128));
			col.alpha = 0 ; 
			set_pixel_color(image,i,j,col.rgb_to_int()); 
		}

	}

delete maxmin ; 
}


void ImageManager::set_contrast(SDL_Surface* image){
	const int val = 200 ; 
	for(int i = 0 ; i < image->w;i++){
		for(int j = 0 ; j<image->h;j++){
			RGB col = get_pixel_color(image,i,j);
			col.red=col.red<=val ? 0 : 255 ;
			col.blue=col.blue<=val? 0 : 255;
			col.green=col.green<=val? 0 : 255 ;
			set_pixel_color(image,i,j,col.rgb_to_int());

		}

	}


}





/***************************************************************************************************************/


void ImageManager::set_contrast_sigmoid(SDL_Surface *image,int threshold){
	
	for(int i = 0 ; i < image->w ; i++){
		for(int j = 0 ; j < image->h ; j++){
			RGB color = get_pixel_color(image,i,j);
			//RGB normalized = normalize_0_1(color);


		}

	}



}




/***************************************************************************************************************/


constexpr double radiant_to_degree(double rad){

	return rad*180.f/M_PI; 

}

constexpr double get_pixel_height(double color_component){
	
	return (255 - color_component);

}
void ImageManager::calculate_normal_map(SDL_Surface* surface,double fact,Uint8 greyscale){
	
	set_greyscale_average(surface,greyscale);
	int height = surface->h;
	int width = surface->w;
	RGB** data = new RGB*[width];
	for(int i = 0 ; i < width ; i++)
		data[i]= new RGB[height];



	for(int i = 0 ; i < width ; i++){
		for(int j = 0 ; j < height ; j++){
			data[i][j] = get_pixel_color(surface,i,j);
		}
	}


	for(int i = 0 ; i < width ; i++){
		for(int j = 0 ; j < height ; j++){

			int x , y , a , b ; 
			x = (i == 0 ) ? i+1 : i-1 ;
			y = (j == 0 ) ? j+1 : j-1 ;
			a = (i == width-1) ? width-2 : i+1;
			b = (j == height-1)? height-2 :j+1 ; 


			double col_left = data[i][y].green;
			double col_right = data[i][b].green;
			double col_up = data[x][j].green;
			double col_down = data[a][j].green;
			double col_up_right = data[x][b].green;
			double col_up_left = data[x][y].green;
			double col_down_left = data[a][y].green;
			double col_down_right = data[a][b].green;
			float atten = 1.56f ; 
			double dx = atten*(fact*(col_right - col_left)/255);
			double dy = atten*(fact*(col_up - col_down)/255) ;
			double ddx = atten*(fact*(col_up_right - col_down_left)/255);
			double ddy = atten*(fact*(col_up_left - col_down_right)/255) ;
			auto Nx = normalize(-1,1,lerp(dy , ddy , 0.5)) ; 
			auto Ny = normalize(-1,1,lerp(dx , ddx , 0.5)) ; 
			auto Nz = 255.0 ; //the normal vector
			
			
			RGB col = RGB((int) floor(truncate(Nx)) , (int) floor(truncate(Ny)) , (int)Nz) ;
			set_pixel_color(surface,i,j,col.rgb_to_int()); 
			


		}

	}	
	

	for(int i = 0 ; i < width ; i++)
		delete[] data[i];
	delete[] data; 

}








/***************************************************************************************************************/




void ImageManager::compute_dudv(SDL_Surface* surface,double factor){
	
	int height = surface->h;
	int width = surface->w;
	RGB** data = new RGB*[width];
	for(int i = 0 ; i < width ; i++)
		data[i]= new RGB[height];



	for(int i = 0 ; i < width ; i++){
		for(int j = 0 ; j < height ; j++){
			data[i][j] = get_pixel_color(surface,i,j);
		}
	}


	for(int i = 0 ; i < width ; i++){
		for(int j = 0 ; j < height ; j++){

			int x , y , a , b ; 
			x = (i == 0 ) ? i+1 : i-1 ;
			y = (j == 0 ) ? j+1 : j-1 ;
			a = (i == width-1) ? width-2 : i+1;
			b = (j == height-1)? height-2 :j+1 ; 


			RGB col_left = data[i][y];
			RGB col_right = data[i][b];
			RGB col_up = data[x][j];
			RGB col_down = data[a][j];
			RGB col_up_right = data[x][b];
			RGB col_up_left = data[x][y];
			RGB col_down_left = data[a][y];
			RGB col_down_right = data[a][b];
			double atten = 0.8 ; 
			double dx_red = atten*(factor*(col_left.red - col_right.red)/255) ; 
			double dx_green = atten*(factor*(col_left.green - col_right.green)/255);	
			double dy_red = atten*(factor*(col_up.red - col_down.red)/255);
			double dy_green = atten*(factor*(col_up.green - col_down.green)/255) ; 

			double ddx_green = atten*(factor*(col_up_right.green - col_down_left.green)/255);
			double ddy_green = atten*(factor*(col_up_left.green - col_down_right.green)/255) ;
			double ddx_red = atten*(factor*(col_up_right.red - col_down_left.red)/255);
			double ddy_red = atten*(factor*(col_up_left.red - col_down_right.red)/255) ;

			
			auto red_var =normalize(-1,1, lerp(dx_red+dy_red, ddx_red+ddy_red,0.5));			
			auto green_var = normalize(-1,1,lerp(dx_green+dy_green , ddx_green+ddy_green,0.5));


			RGB col = RGB( truncate(red_var) ,  truncate(green_var) , 0.0) ; 
			set_pixel_color(surface,i,j,col.rgb_to_int()); 
			


		}

	}	
	

	for(int i = 0 ; i < width ; i++)
		delete[] data[i];
	delete[] data;



}















































}
