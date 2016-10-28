#include <iostream>
#include <cstdlib>
#include "../includes/ImageManager.h"
#include "../includes/ImageImporter.h"
#include "../includes/Window.h"
#include "../includes/EventHandler.h"
using namespace std;
using namespace maptomix;
void init_api(){
	if(SDL_Init(SDL_INIT_EVERYTHING)<0)
	{
		cout<<"SDL_Init problem : "<<SDL_GetError() <<endl; 

	}
	if(!IMG_Init(IMG_INIT_JPG|IMG_INIT_PNG)) {

		cout<<"IMG init problem : " <<IMG_GetError()<<endl;
	}	

	

}





int main(int argv , char** argc){
	init_api();

	Window *window = new Window(400,400,"ok");
	SDL_Event event; 
	EventHandler *handler=EventHandler::getInstance(); 
	handler->setEvent(event);
	ImageImporter *importer = ImageImporter::getInstance();
	SDL_Surface *s = importer ->load_image(argc[1]);
	window->display_image(s);
	uint32_t color = ImageManager::get_pixel_color(s,39,38);
	ImageManager::print_pixel(color);
	handler->main_loop();



delete window;

handler->close(); 
importer->close(); 
IMG_Quit();
SDL_Quit();
return EXIT_SUCCESS ; 
}
