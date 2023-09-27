#include "../includes/ImageManager.h"

#include <iostream>
#include <assert.h>
#include <thread>
#include <cstdlib>
#include <climits>
#include <string>
#include <ctime>
#include <future>

namespace axomae{

using namespace std;
bool ImageManager::gpu = false;
static bool CHECK_IF_CUDA_AVAILABLE() {
	if (!ImageManager::USING_GPU()) 
		return false; 
	else 
		return true;
}

template<typename T> 
static void replace_image(SDL_Surface* surface, T* image, unsigned int size, int bpp);
RGB::RGB():red(0.),green(0.),blue(0.),alpha(0.){}
RGB::RGB(float r , float g , float b):red(r),green(g),blue(b),alpha(0){}
RGB::RGB(float r , float g , float b , float a):red(r),green(g),blue(b),alpha(a){}
RGB::~RGB(){}
ImageManager::ImageManager(){}
ImageManager::~ImageManager(){} 

/**************************************************************************************************************/
template <typename T> 
uint8_t truncate(T n){
	if(static_cast<float> (n) <= 0.)
		return 0;
	else if (static_cast<float>(n) >= 255)
		return 255;
	else
		return static_cast<float> (n) ; 
}

/**************************************************************************************************************/
RGB ImageManager::get_pixel_color(SDL_Surface* surface , int x , int y ){
	int bpp = surface->format->BytesPerPixel;
	uint8_t *color = (uint8_t*) (surface->pixels) + x*bpp + y*surface->pitch;
	float red = 0 , blue = 0 , green = 0 , alpha = 0 ; 
	switch(bpp){
		case 1 :
			{
				red = *color >> 5 & 0x7;
				green = *color >> 2 & 0x7;
				blue = *color & 0x3 ;
			}
		break;
		case 2:
			{
				uint16_t colo16bits = *(uint16_t*)color;
				if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
					red = colo16bits >> 12 & 0xF;
					green = colo16bits >> 8 & 0XF;
					blue = colo16bits >> 4 & 0XF;
					alpha = colo16bits & 0XF;
				}
				else {
					alpha = colo16bits >> 12 & 0xF;
					blue = colo16bits >> 8 & 0XF;
					green = colo16bits >> 4 & 0XF;
					red = colo16bits & 0XF;
				}
			}
		break;
		case 3:
			{
				uint32_t colo24bits = *(uint32_t*)color;
				if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
					red = colo24bits >> 16 & 0XFF;
					green = colo24bits >> 8 & 0XFF;
					blue = colo24bits & 0XFF;
				}
				else {
					blue = colo24bits >> 16 & 0XFF;
					green = colo24bits >> 8 & 0XFF;
					red = colo24bits & 0XFF;
				}
			}
		break;
		case 4:
			{
				uint32_t colo32bits = *(uint32_t*) color ; 
				if(SDL_BYTEORDER == SDL_BIG_ENDIAN){
					red = colo32bits >> 24 & 0XFF;
					green = colo32bits >> 16 & 0XFF;
					blue = colo32bits >> 8 & 0XFF;
					alpha = colo32bits & 0XFF ; 
				}
				else{
					alpha = colo32bits >> 24 & 0XFF;
					blue = colo32bits >> 16 & 0XFF;
					green = colo32bits >> 8 & 0XFF;
					red = colo32bits & 0XFF ; 
				}
			}
		break;
		}
	RGB rgb = RGB(static_cast<float> (red) ,static_cast<float> (green) ,static_cast<float> (blue) ,static_cast<float> (alpha)) ; 
	return rgb;
}

/**************************************************************************************************************/
void ImageManager::set_pixel_color(SDL_Surface* surface, int x, int y, uint32_t color) {
	//TODO : Add Tiling management when uv coordinates greater than limit
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
	for(int i = 0 ; i < w ; i++)
		for(int j = 0 ; j < h ; j++)
			set_pixel_color(surface,i,j,arrayc[i][j].rgb_to_int());
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
	if (CHECK_IF_CUDA_AVAILABLE()) 
		GPU_compute_greyscale(image, false); 
	else 
		for (int i = 0; i < image->w; i++) 
			for (int j = 0; j < image->h; j++) {
				RGB rgb = get_pixel_color(image, i, j);
				rgb.red = (rgb.red + rgb.blue + rgb.green) / factor;
				rgb.green = rgb.red;
				rgb.blue = rgb.red;
				uint32_t gray = rgb.rgb_to_int();
				set_pixel_color(image, i, j, gray);
			}		
}

/**************************************************************************************************************/
static void replace_image(SDL_Surface* surface, uint8_t* image) {
	int bpp = surface->format->BytesPerPixel; 
	SDL_LockSurface(surface);
 	if (bpp == 1) {
		for ( int i = 0; i < surface->w; i++)
			for ( int j = 0; j < surface->h; j++)
				 ((Uint8*)surface->pixels)[i*bpp + j*surface->pitch] = image[i*bpp + j*surface->pitch];
		delete[] static_cast<uint8_t*> (image);
	}
	else 
		std::cerr << "error reading image ... BPP is : " << std::to_string(bpp) << " Bytes per pixel\n";
	SDL_UnlockSurface(surface);
}


/**************************************************************************************************************/
static void replace_image(SDL_Surface* surface, uint16_t* image) {
	int bpp = surface->format->BytesPerPixel;
	SDL_LockSurface(surface);
 	if (bpp == 2) {
		for ( int i = 0; i < surface->w; i++)
			for ( int j = 0; j < surface->h; j++)
				((Uint16*)surface->pixels)[i*bpp + j*surface->pitch] = image[i*bpp + j*surface->pitch];
		delete[]  static_cast<uint16_t*> (image);
	}		
	else 
		std::cerr << "error reading image ... BPP is : " << std::to_string(bpp) << " Bytes per pixel\n";
	SDL_UnlockSurface(surface);
}


/**************************************************************************************************************/

static void replace_image(SDL_Surface* surface, uint32_t* image) {
	int bpp = surface->format->BytesPerPixel; 
	int pitch = surface->pitch; 
	SDL_LockSurface(surface);
	if (bpp == 4) {
		for ( int i = 0; i < surface->w; i++)
			for ( int j = 0; j < surface->h; j++) 
				((Uint32*)surface->pixels)[i*bpp + j*surface->pitch] = image[i*bpp + j*surface->pitch];	
		delete[] static_cast<uint32_t*> (image);

	}
	else if (bpp == 3) 
		for ( int i = 0; i < surface->w; i++)
			for ( int j = 0; j < surface->h; j++)
				((Uint32*)surface->pixels)[i*bpp + j*surface->pitch] = image[i*bpp + j*surface->pitch];
	else 
		std::cerr << "error reading image ... BPP is : " << std::to_string(bpp) << " Bytes per pixel\n";	
	SDL_UnlockSurface(surface);
}

/**************************************************************************************************************/
void ImageManager::set_greyscale_luminance(SDL_Surface* image){
	bool cuda = CHECK_IF_CUDA_AVAILABLE() ; 
	std::clock_t clock;
	if (cuda) {
		GPU_compute_greyscale(image, true);
	}	
	else {
		assert(image != nullptr);
		for (int i = 0; i < image->w; i++) 
			for (int j = 0; j < image->h; j++) {
				RGB rgb = get_pixel_color(image, i, j);
				rgb.red = floor(rgb.red*0.3 + rgb.blue*0.11 + rgb.green*0.59);
				rgb.green = rgb.red;
				rgb.blue = rgb.red;
				uint32_t gray = rgb.rgb_to_int();
				set_pixel_color(image, i, j, gray);
			}
	}
}



/**************************************************************************************************************/
double RGB::intensity(){
	double av=(red+green+blue)/3;
	return av/255;
}

/**************************************************************************************************************/

void RGB::clamp(){
	if(red > 255) 
		red = 255 ; 
	if(green > 255)
		green = 255 ; 
	if(blue > 255) 
		blue = 255 ;
	if(red < 0)
		red = 0 ; 
	if(green < 0) 
		green = 0 ; 
	if(blue < 0)
		blue = 0 ; 

}




/**************************************************************************************************************/
RGB RGB::int_to_rgb(uint32_t val){
	RGB rgb = RGB(); 
	if(SDL_BYTEORDER == SDL_BIG_ENDIAN){
		rgb.red   = static_cast<float> (val >> 24 & 0XFF) ; 
		rgb.green = static_cast<float> (val >> 16 & 0XFF) ; 
		rgb.blue  = static_cast<float> (val >> 8 & 0XFF) ; 
		rgb.alpha = static_cast<float> (val & 0XFF);
	}
	else{
		rgb.alpha = static_cast<float> (val >> 24 & 0XFF) ; 
		rgb.blue  = static_cast<float> (val >> 16 & 0XFF) ; 
		rgb.green = static_cast<float> (val >> 8 & 0XFF) ; 
		rgb.red   = static_cast<float> (val & 0XFF);
	}
	return rgb; 
}

/**************************************************************************************************************/
RGB RGB::int_to_rgb(uint16_t val){
	RGB rgb = RGB(); 
	if(SDL_BYTEORDER == SDL_BIG_ENDIAN){
		rgb.red   = static_cast<float> (val >> 12 & 0XF) ; 
		rgb.green = static_cast<float> (val >> 8 & 0XF) ; 
		rgb.blue  = static_cast<float> (val >> 4 & 0XF) ; 
		rgb.alpha = static_cast<float> (val & 0XF);
	}
	else{
		rgb.alpha = static_cast<float> (val >> 12 & 0XF) ; 
		rgb.blue  = static_cast<float> (val >> 8 & 0XF) ; 
		rgb.green = static_cast<float> (val >> 4 & 0XF) ; 
		rgb.red   = static_cast<float> (val & 0XF);
	}
	return rgb; 
}

/**************************************************************************************************************/
uint32_t RGB::rgb_to_int(){
	uint32_t image=0 ;
	int b = static_cast<int> (std::round(blue))  , g = static_cast<int> (std::round(green)) , r = static_cast<int> (std::round(red)) , a = static_cast<int> (std::round(alpha)) ; 
	if(SDL_BYTEORDER == SDL_BIG_ENDIAN)
		image = a | (b << 8) | (g << 16) | (r << 24);	
	else
		image = r | (g << 8) | (b << 16) | (a << 24);	
	return image ; 
}

/**************************************************************************************************************/
template<typename T>
static T compute_kernel_pixel(RGB **data,const T kernel[3][3],int i,int j,uint8_t flag){
	if(flag == AXOMAE_RED){
		return  data[i-1][j-1].red*kernel[0][0]+data[i][j-1].red*kernel[0][1]+data[i+1][j-1].red*kernel[0][2]+
			 data[i-1][j].red*kernel[1][0] + data[i][j].red*kernel[1][1]+data[i+1][j].red*kernel[1][2]+
			 data[i-1][j+1].red*kernel[2][0]+ data[i][j+1].red*kernel[2][1]+data[i+1][j+1].red*kernel[2][2];			
	}
	else if (flag==AXOMAE_BLUE){
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
template<typename T>
static RGB compute_generic_kernel_pixel(const RGB **data ,const unsigned int size ,const T **kernel , const unsigned int x , const unsigned int y){
	float r = 0 , g = 0 , b = 0 ; 
	uint8_t middle = std::floor(size / 2) ; 
	for(unsigned int i = 0 ; i < size ; i ++)
		for(unsigned int j = 0 ; j < size ; j ++){
			float kernel_val = kernel[i][j] ; 
			int new_i = i - middle , new_j = j - middle ;
			r += data[x + new_i][y + new_j].red * kernel_val ;
			g += data[x + new_i][y + new_j].green * kernel_val ; 
			b += data[x + new_i][y + new_j].blue * kernel_val ;
		}	
	return  RGB(r ,g ,b) ; 
}

/**************************************************************************************************************/
void ImageManager::compute_edge(SDL_Surface* surface,uint8_t flag,uint8_t border ){
	bool cuda = CHECK_IF_CUDA_AVAILABLE();
	if  (cuda) 
		GPU_compute_height(surface, flag, border);
	else  {
		//TODO : use multi threading for initialization and greyscale computing
		/*to avoid concurrent access on image*/

		std::cout << "CUDA : " << cuda << "\n"; 

		int w = surface->w;
		int h = surface->h;
		//thread this :
		RGB **data = new RGB*[w];
		for (int i = 0; i < w; i++)
			data[i] = new RGB[h];
		float max_red = 0, max_blue = 0, max_green = 0;
		float min_red = 0, min_blue = 0, min_green = 0;
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
				if (flag == AXOMAE_USE_SOBEL) {
					arr_v[i][j] = sobel_mask_vertical[i][j];
					arr_h[i][j] = sobel_mask_horizontal[i][j];
				}
				else if (flag == AXOMAE_USE_PREWITT) {
					arr_h[i][j] = prewitt_mask_horizontal[i][j];
					arr_v[i][j] = prewitt_mask_vertical[i][j];
				}
				else if (flag == AXOMAE_USE_SCHARR) {
					arr_h[i][j] = scharr_mask_horizontal[i][j];
					arr_v[i][j] = scharr_mask_vertical[i][j];
				}
			}
		}
		for (int i = 1; i < w - 1; i++) {
			for (int j = 1; j < h - 1; j++) {
				if (border == AXOMAE_REPEAT) {
					float setpix_h_red = 0, setpix_v_red = 0, setpix_h_green = 0, setpix_v_green = 0, setpix_v_blue = 0, setpix_h_blue = 0;
					setpix_h_red = compute_kernel_pixel(data, arr_h, i, j, AXOMAE_RED);
					setpix_v_red = compute_kernel_pixel(data, arr_v, i, j, AXOMAE_RED);
					setpix_h_green = compute_kernel_pixel(data, arr_h, i, j, AXOMAE_GREEN);
					setpix_v_green = compute_kernel_pixel(data, arr_v, i, j, AXOMAE_GREEN);
					setpix_h_blue = compute_kernel_pixel(data, arr_h, i, j, AXOMAE_BLUE);
					setpix_v_blue = compute_kernel_pixel(data, arr_v, i, j, AXOMAE_BLUE);
					setpix_v_red = normalize(max_red, min_red, setpix_v_red);
					setpix_h_red = normalize(max_red, min_red, setpix_h_red);
					setpix_v_green = normalize(max_green, min_green, setpix_v_green);
					setpix_h_green = normalize(max_green, min_green, setpix_h_green);
					setpix_v_blue = normalize(max_blue, min_blue, setpix_v_blue);
					setpix_h_blue = normalize(max_blue, min_blue, setpix_h_blue);
					float r = magnitude(setpix_v_red, setpix_h_red);
					float g = magnitude(setpix_v_green, setpix_h_green);
					float b = magnitude(setpix_v_blue, setpix_h_blue);
					RGB rgb = RGB(r, g, b, 0);
					set_pixel_color(surface, i, j, rgb.rgb_to_int());
				}
			}
		}
		auto del = std::async(std::launch::async, [data, w]() {
			for (int i = 0; i < w; i++)
				delete[] data[i];
			delete[] data;
		});

	}
}

/**************************************************************************************************************/



template<typename T>
RGB RGB::operator*(T arg) const {
	RGB rgb = RGB(static_cast<float>(red * arg) , static_cast<float> (green * arg) , static_cast<float> (blue * arg) , static_cast<float> (alpha * arg)); 
	return rgb ; 
}


RGB RGB::operator+=(float arg) const {
	RGB rgb=RGB(red+arg,green+arg,blue+arg,alpha+arg);
	return rgb;
	
}

RGB RGB::operator+=(RGB arg) const {
	RGB rgb=RGB(red+arg.red,green+arg.green,blue+arg.blue,alpha+arg.alpha);
	return rgb;
	
}

RGB RGB::operator+(RGB arg) const {
	RGB rgb = RGB(red+arg.red,green+arg.green,blue+arg.blue,alpha+arg.alpha);
	return rgb;
}

RGB RGB::operator/(float arg) const {
	assert(arg>0);
	RGB rgb = RGB(red/arg , green/arg, blue/arg,alpha/arg);
	return rgb;
}

RGB RGB::operator-(RGB arg) const {
	RGB rgb = RGB(red - arg.red , green - arg.green , blue - arg.blue); 
	return rgb ; 
}

/**************************************************************************************************************/
void RGB::invert_color(){
	red=abs(red-255);
  	green=abs(green-255) ;
 	blue=abs(blue-255);
  	alpha=abs(alpha-255);
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

/**************************************************************************************************************/
void ImageManager::set_contrast(SDL_Surface* image,int level){
	double correction_factor = (259*(level+255))/(255*(259-level));
	max_colors *maxmin = get_colors_max_variation(image);
	for(int i = 0 ; i < image->w; i++){
		for(int j = 0 ; j < image->h; j++){
			RGB col = get_pixel_color(image,i,j);
			col.red =  floor(truncate(correction_factor * (col.red - 128)+128));
			col.green =  floor(truncate(correction_factor * (col.green - 128)+128));
			col.blue =  floor(truncate(correction_factor * (col.blue - 128)+128));
			col.alpha = 0 ; 
			set_pixel_color(image,i,j,col.rgb_to_int()); 
		}
	}
	delete maxmin ; 
}

/**************************************************************************************************************/
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
/*TODO : Contrast image enhancement */
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

/**************************************************************************************************************/
constexpr double get_pixel_height(double color_component){
	return (255 - color_component);
}

/**************************************************************************************************************/
void ImageManager::compute_normal_map(SDL_Surface* surface,double fact , float attenuation){
	bool cuda = CHECK_IF_CUDA_AVAILABLE(); 
	if (cuda)
		GPU_compute_normal(surface, fact , AXOMAE_REPEAT);
	else {
		int height = surface->h;
		int width = surface->w;
		RGB** data = new RGB*[width];
		for (int i = 0; i < width; i++)
			data[i] = new RGB[height];
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++)
				data[i][j] = get_pixel_color(surface, i, j);
		}
		for (int i = 0; i < width; i++) {
			for (int j = 0; j < height; j++) {
				int x, y, a, b;
				x = (i == 0) ? i + 1 : i - 1;
				y = (j == 0) ? j + 1 : j - 1;
				a = (i == width - 1) ? width - 2 : i + 1;
				b = (j == height - 1) ? height - 2 : j + 1;
				double col_left = data[i][y].green;
				double col_right = data[i][b].green;
				double col_up = data[x][j].green;
				double col_down = data[a][j].green;
				double col_up_right = data[x][b].green;
				double col_up_left = data[x][y].green;
				double col_down_left = data[a][y].green;
				double col_down_right = data[a][b].green;
				float atten = attenuation;
				double dx = atten*(fact*(col_right - col_left) / 255);
				double dy = atten*(fact*(col_up - col_down) / 255);
				double ddx = atten*(fact*(col_up_right - col_down_left) / 255);
				double ddy = atten*(fact*(col_up_left - col_down_right) / 255);
				auto Nx = normalize(-1, 1, lerp(dy, ddy, 0.5));
				auto Ny = normalize(-1, 1, lerp(dx, ddx, 0.5));
				auto Nz = 255.0; //the normal vector
				RGB col = RGB(floor(truncate(Nx)), floor(truncate(Ny)), Nz);
				set_pixel_color(surface, i, j, col.rgb_to_int());
			}
		}
		auto del = std::async(std::launch::async, [data, width, height]() {
			for (int i = 0; i < width; i++)
				delete[] data[i];
			delete[] data;
		});
	}
}

/***************************************************************************************************************/
void ImageManager::compute_dudv(SDL_Surface* surface,double factor){
	int height = surface->h;
	int width = surface->w;
	RGB** data = new RGB*[width];
	for(int i = 0 ; i < width ; i++)
		data[i]= new RGB[height];
	for(int i = 0 ; i < width ; i++){
		for(int j = 0 ; j < height ; j++)
			data[i][j] = get_pixel_color(surface,i,j);
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
	auto del = std::async(std::launch::async, [data, width, height]() {
		for (int i = 0; i < width; i++)
			delete[] data[i];
		delete[] data;
	}); 
}

/***************************************************************************************************************/
float bounding_coords(float x , float y , float z , bool min) { 
	if(min){
		if( x <= y )
			return x <= z ? x : z ; 
		else
			return y <= z ? y : z ; 
	}
	else{
		if(x >= y)
			return x >= z ? x : z ; 
		else
			return y >= z ? y : z ; 
	}		
}

/***************************************************************************************************************/
/* Get barycentric coordinates of I in triangle P1P2P3 */
Vect3D barycentric_lerp(Point2D P1 , Point2D P2 , Point2D P3 , Point2D I){
		float W1 = (( P2.y - P3.y ) * ( I.x - P3.x ) + ( P3.x - P2.x ) * ( I.y - P3.y )) / (( P2.y - P3.y ) * ( P1.x - P3.x ) + ( P3.x - P2.x ) * ( P1.y - P3.y )) ; 
		float W2 = (( P3.y - P1.y ) * ( I.x - P3.x ) + ( P1.x - P3.x ) * ( I.y - P3.y )) / (( P2.y - P3.y ) * ( P1.x - P3.x ) + ( P3.x - P2.x ) * ( P1.y - P3.y )) ; 
		float W3 = 1 - W1 - W2 ; 
		Vect3D v = {W1 , W2 , W3} ; 
		return v ;  
}

/***************************************************************************************************************/
Vect3D tan_space_transform(Vect3D T , Vect3D BT , Vect3D N , Vect3D I){
		Vect3D result = { I.x * BT.x + I.y * BT.y + I.z * BT.z , I.x * T.x + T.y * I.y + T.z * I.z ,I.x * N.x + N.y * I.y + I.z * N.z};
		return result ; 
}

/***************************************************************************************************************/
uint32_t compute_normals_set_pixels_rgb(  Point2D P1 , 
					  Point2D P2 , 
					  Point2D P3 , 
					  Vect3D N1 ,
					  Vect3D N2 , 
					  Vect3D N3 , 
					  Vect3D BT1 , 
					  Vect3D BT2 , 
					  Vect3D BT3 ,
					  Vect3D T1 , 
					  Vect3D T2 , 
					  Vect3D T3 , 
					  int x , 
					  int y ,  
					  int x_min , 
					  int y_min , 
					  int x_max , 
					  int y_max ,
					  bool tangent_space){

	Point2D I = {static_cast<float>(x) ,static_cast<float> (y)} ; 
	Vect3D C = barycentric_lerp(P1 , P2 , P3 , I) ;
	if(C.x >= 0 && C.y >= 0 && C.z >= 0){
		auto interpolate = [&C](Vect3D N1 , Vect3D N2 , Vect3D N3){
			Vect3D normal = { N1.x * C.x + N2.x * C.y + N3.x * C.z , N1.y * C.x + N2.y * C.y + N3.y * C.z ,N1.z * C.x + N2.z * C.y + N3.z * C.z }; 		
			return normal ; 
		};
		if(tangent_space){
			Vect3D normal = interpolate(N1 , N2 , N3) ; 	
			Vect3D B = {(BT1.x + BT2.x + BT3.x)/3 , (BT1.y + BT2.y + BT3.y)/3 , (BT1.z + BT2.z + BT3.z)/3 }; 
			Vect3D T = {(T1.x + T2.x + T3.x)/3 , (T1.y + T2.y + T3.y)/3 , (T1.z + T2.z + T3.z)/3 }; 
			Vect3D N = {(N1.x + N2.x + N3.x)/3 , (N1.y + N2.y + N3.y)/3 , (N1.z + N2.z + N3.z)/3 }; 
			B.normalize(); 
			T.normalize(); 
			N.normalize(); 
			normal = tan_space_transform(T , B , N , normal);//TODO : provide a better algorithm for tang space  
			normal.normalize() ; 	
			RGB rgb = RGB( (normal.x * 255 + 255)/2 ,(normal.y * 255+255)/2 , (normal.z * 255+255)/2 , 0 ) ; 
			uint32_t val = rgb.rgb_to_int() ; 
			return val;   
		}
		else{
			N1.normalize(); 
			N2.normalize(); 
			N3.normalize(); 
			Vect3D normal = interpolate(N1 , N2 , N3) ; 
			RGB rgb = RGB((normal.x * 255 + 255)/2 ,(normal.y * 255+255)/2 , (normal.z * 255+255)/2 , 0) ; 
			uint32_t val = rgb.rgb_to_int() ;
			return val; 
		}
	}
	else{ 
		return 0 ; 
	}
}
/***************************************************************************************************************/
/** @brief Computes the normals at the position of each texel of a 3D model and displays them on the projected UV
 * This function allows us to check the distribution of normals across the mesh.  
 */
SDL_Surface* ImageManager::project_uv_normals(Object3D object , int width ,  int height , bool tangent_space){
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
	Uint32 rmask = 0xFF000000 ; 
	Uint32 gmask = 0x00FF0000 ; 
	Uint32 bmask = 0x0000FF00 ; 
	Uint32 amask = 0x000000FF ; 
#else 
	Uint32 amask = 0xFF000000 ; 
	Uint32 bmask = 0x00FF0000 ; 
	Uint32 gmask = 0x0000FF00 ; 
	Uint32 rmask = 0x000000FF ; 
#endif	

	SDL_Surface* surf = SDL_CreateRGBSurface(0 , width , height , 24 , rmask , gmask , bmask , amask) ;
	assert(surf != nullptr) ; 
	bool tangents_empty = object.tangents.empty()  , 
	     bitangents_empty = object.bitangents.empty() , 
	     normals_empty = object.normals.empty() ,
	     uv_empty = object.uv.empty() , 
	     indices_empty = object.indices.empty(); 
	if(tangents_empty || normals_empty || uv_empty || indices_empty)
		return surf ; 
	/* TODO : Parallelize : each face = 1 thread ? */
	for(unsigned int i = 0 ; i < object.indices.size() ; i+=3) {
		std::vector<unsigned int> index = object.indices; 
		Point2D P1 = {object.uv[index[i] * 2] , object.uv[index[i] * 2 + 1]} ; 	
		Point2D P2 = {object.uv[index[i + 1] * 2] , object.uv[index[i + 1] * 2 + 1]} ; 
		Point2D P3 = {object.uv[index[i + 2] * 2] , object.uv[index[i + 2] * 2 + 1]} ; 
		Vect3D N1  = {object.normals[index[i] * 3] , object.normals[index[i] * 3 + 1] , object.normals[index[i] * 3 + 2]} ; 
		Vect3D N2  = {object.normals[index[i + 1] * 3] , object.normals[index[i + 1] * 3 + 1] , object.normals[index[i + 1] * 3 + 2]} ; 
		Vect3D N3  = {object.normals[index[i + 2] * 3] , object.normals[index[i + 2] * 3 + 1] , object.normals[index[i + 2] * 3 + 2]} ; 
		Vect3D BT1 = {object.bitangents[index[i] * 3] , object.bitangents[index[i] * 3 + 1] , object.bitangents[index[i] * 3 + 2]} ; 
		Vect3D BT2 = {object.bitangents[index[i + 1] * 3]  , object.bitangents[index[i + 1] * 3 + 1] , object.bitangents[index[i + 1] * 3 + 2]};
		Vect3D BT3 = {object.bitangents[index[i + 2] * 3] , object.bitangents[index[i + 2] * 3 + 1] , object.bitangents[index[i + 2] * 3 + 2]};
		Vect3D T1  = {object.tangents[index[i] * 3] , object.tangents[index[i] * 3 + 1] , object.tangents[index[i] * 3 + 2]} ; 
		Vect3D T2  = {object.tangents[index[i + 1] * 3]  , object.tangents[index[i + 1] * 3 + 1] , object.tangents[index[i + 1] * 3 + 2]} ; 
		Vect3D T3  = {object.tangents[index[i + 2] * 3] , object.tangents[index[i + 2] * 3 + 1] , object.tangents[index[i + 2] * 3 + 2]} ; 
		auto clamp_uv = [&](Point2D P) { /*check if UVs are correctly set in bounds ... if not , we use modulus*/
			if (P.x > 1.f)
				P.x = std::fmod(P.x , 1.f) ; 
			if (P.y > 1.f) 
				P.y = std::fmod(P.y , 1.f) ; 
			return P ; 
		};
		P1 = clamp_uv(P1) ; 
		P2 = clamp_uv(P2) ; 
		P3 = clamp_uv(P3) ; 
		P1.x *= width ; 
		P1.y *= height ; 
		P2.x *= width ; 
		P2.y *= height ; 
		P3.x *= width ; 
		P3.y *= height ; 
		int x_max = static_cast<int>(bounding_coords( P1.x , P2.x , P3.x , false)); 
		int x_min = static_cast<int>(bounding_coords( P1.x , P2.x , P3.x , true )); 
		int y_max = static_cast<int>(bounding_coords( P1.y , P2.y , P3.y , false)); 
		int y_min = static_cast<int>(bounding_coords( P1.y , P2.y , P3.y , true)); 
		for(int x = x_min ; x <= x_max ; x++)
			for(int y = y_min ; y <= y_max ; y++){
				uint32_t val = compute_normals_set_pixels_rgb( P1 , P2 , P3 , N1 , N2 , N3 , BT1 , BT2 , BT3 , T1 , T2 , T3 , x , y , x_min , y_min , x_max , y_max , tangent_space); 
				if (val != 0)
					set_pixel_color(surf , x , y , val) ;
			}
	}
	return surf ; 
} 
/**************************************************************************************************************/
void ImageManager::smooth_image(SDL_Surface* surface , FILTER filter , const unsigned int factor){
	if(surface!= nullptr){
		int height = surface->h;
		int width = surface->w;
		RGB** data = new RGB*[width];
		for(int i = 0 ; i < width ; i++)
			data[i]= new RGB[height];
		for(int i = 0 ; i < width ; i++){
			for(int j = 0 ; j < height ; j++)
				data[i][j] = get_pixel_color(surface,i,j);
		}
		int n = 0 ;
		void* convolution_kernel = nullptr ; 
		switch (filter) {
			case GAUSSIAN_SMOOTH_3_3:
				convolution_kernel = (void*) gaussian_blur_3_3 ; 
				n = 3 ;
			break ; 
			case GAUSSIAN_SMOOTH_5_5:
				convolution_kernel = (void*) gaussian_blur_5_5 ; 
				n = 5 ; 
			break ; 
			case BOX_BLUR:
				convolution_kernel = (void*) box_blur ;  
				n = 3 ; 
			break ; 
			default : 
				convolution_kernel = nullptr ; 
				n = 0 ; 
			break ; 
		}
		float **blur = new float*[n] ;
		static unsigned int kernel_index = 0 ; 
		for(int i = 0 ; i < n ; i++)
			blur[i] = new float[n]; 
		for(int i = 0 ; i < n ; i++)
			for(int j = 0 ; j < n ; j++ , kernel_index++)
				blur[i][j]=static_cast<float*>(convolution_kernel)[kernel_index] ;
		kernel_index = 0 ;
		unsigned int middle = std::floor(n/2) ; 	
			for(unsigned int i = middle ; i < width-middle ; i++){
				for(unsigned int j = middle ; j < height-middle ; j++){	
					RGB col;
					col = compute_generic_kernel_pixel(const_cast<const RGB**> (data) , n , const_cast<const float**>(blur)  , i , j) ; 
					set_pixel_color(surface,i,j,col.rgb_to_int()); 
				}
			}
		for(int i = 0 ; i < n ; i++)
			delete[] blur[i] ; 
		delete[] blur ; 

		auto del = std::async(std::launch::async, [data, width, height]() {
		for (int i = 0; i < width; i++)
			delete[] data[i];
		delete[] data;
		}); 
		if(factor != 0)
			smooth_image(surface , filter , factor - 1); 
	}
}
/**************************************************************************************************************/
void ImageManager::sharpen_image(SDL_Surface* surface , FILTER filter , const unsigned int factor){
	if(surface!= nullptr){
		int height = surface->h;
		int width = surface->w;
		RGB** data = new RGB*[width];
		for(int i = 0 ; i < width ; i++)
			data[i]= new RGB[height];
		for(int i = 0 ; i < width ; i++)
			for(int j = 0 ; j < height ; j++)
				data[i][j] = get_pixel_color(surface,i,j);
		
		int n = 0 ;
		void* convolution_kernel = nullptr ; 
		switch (filter) {
			case SHARPEN:
				convolution_kernel = (void*) sharpen_kernel ; 
				n = 3 ;
			break ; 
			case UNSHARP_MASKING:
				convolution_kernel = (void*) unsharp_masking ; 
				n = 5 ; 
			break ; 
			default : 
				convolution_kernel = nullptr ; 
				n = 0 ; 
			break ; 
		}
		float **sharp = new float*[n] ;
		static unsigned int kernel_index = 0 ; 
		for(int i = 0 ; i < n ; i++)
			sharp[i] = new float[n]; 
		for(int i = 0 ; i < n ; i++)
			for(int j = 0 ; j < n ; j++ , kernel_index++)
				sharp[i][j]=static_cast<float*>(convolution_kernel)[kernel_index] ;
		kernel_index = 0 ;
		unsigned int middle = std::floor(n/2) ; 	
			for(unsigned int i = middle ; i < width-middle ; i++){
				for(unsigned int j = middle ; j < height-middle ; j++){	
					RGB col;
					col = compute_generic_kernel_pixel(const_cast<const RGB**> (data) , n , const_cast<const float**>(sharp)  , i , j) ; 
					col.clamp(); 
					set_pixel_color(surface,i,j,col.rgb_to_int());								
				}
			}	
		for(int i = 0 ; i < n ; i++)
			delete[] sharp[i] ; 
		delete[] sharp ; 
		auto del = std::async(std::launch::async, [data, width, height]() {
		for (int i = 0; i < width; i++)
			delete[] data[i];
		delete[] data;
		}); 
	//	if(factor != 0)
	//		sharpen_image(surface , filter , factor - 1); 
	}
}



















//end namespace 
}
