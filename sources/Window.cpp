#include "../includes/Window.h"

using namespace std;
namespace maptomix {
Window::Window(const int w,const int h , const char* n){
	width=w; 
	height=h;
	name =(string) n;


	m_window = SDL_CreateWindow(n,SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,w,h,SDL_WINDOW_SHOWN);

}

Window::~Window(){
	SDL_DestroyWindow(m_window);

}


}
