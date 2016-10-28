#ifndef WINDOW_H
#define WINDOW_H
#include <string>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>


#include "images.h"
#include "View.h"

namespace maptomix{




class Window : public View

	{

public:
	Window(const int width, const int height,const char* name);
	~Window();
        void display_image(SDL_Surface* image);	

private:
	int width;
	int height;
	std::string name;
	SDL_Window *m_window; 
	SDL_Renderer *renderer;
	
	SDL_Texture *texture;
	bool free_surface_texture;


};

}


#endif

