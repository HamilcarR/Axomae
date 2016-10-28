#ifndef WINDOW_H
#define WINDOW_H
#include <string>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

namespace maptomix{

class Window{

public:
	Window(const int width, const int height,const char* name);
	~Window(); 

private:
	int width;
	int height;
	std::string name;
	SDL_Window *m_window; 





};

}


#endif

