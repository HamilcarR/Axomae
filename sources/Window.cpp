#include "../includes/Window.h"
#include <assert.h>
#include <iostream>
namespace maptomix {

using namespace std;

Window::Window(const int w,const int h , const char* n) : maptomix::View(){
	width=w; 
	height=h;
	name =(string) n;


	m_window = SDL_CreateWindow(n,SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,w,h,SDL_WINDOW_SHOWN);
	renderer = SDL_CreateRenderer(m_window,-1,0);
	free_surface_texture = false;
}

Window::~Window(){
	if(free_surface_texture){
	SDL_DestroyTexture(texture);
	
	}
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(m_window);
}

/*****************************************************************************************************************/

void Window::display_image(SDL_Surface* image){
	texture=SDL_CreateTextureFromSurface(renderer,image);
	assert(texture!=nullptr);
	SDL_RenderCopy(renderer,texture,NULL,NULL);
	SDL_RenderPresent(renderer);
	free_surface_texture=true;


}

/*****************************************************************************************************************/
/*****************************************************************************************************************/










/*****************************************************************************************************************/


/*****************************************************************************************************************/


/*****************************************************************************************************************/


/*****************************************************************************************************************/


}
