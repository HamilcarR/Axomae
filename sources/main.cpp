#include <iostream>
#include <cstdlib>
#include "../includes/ImageManager.h"
#include "../includes/ImageImporter.h"
#include "../includes/Window.h"
#include "../includes/EventHandler.h"


using namespace std;
using namespace maptomix;


extern int * contrast = nullptr ; 

void init_api(){
	if(SDL_Init(SDL_INIT_EVERYTHING)<0)
	{
		cout<<"SDL_Init problem : "<<SDL_GetError() <<endl; 

	}
	if(!IMG_Init(IMG_INIT_JPG|IMG_INIT_PNG)) {

		cout<<"IMG init problem : " <<IMG_GetError()<<endl;             
	}	

	

}


void contrast_f(){
	EventHandler* handler = EventHandler::getInstance();
	bool  *arr =handler->get_event_array();
	
	if(arr[UP])
		*contrast++;
	else if (arr[DOWN])
		*contrast--;


}


int main(int argv , char** argc){
	init_api();
	int contr = 0 ; 
        contrast = &contr;	
	Window *window = new Window(600,600,"shet");
	SDL_Event event; 
	EventHandler *handler=EventHandler::getInstance(); 
	handler->setEvent(event);
	ImageImporter *importer = ImageImporter::getInstance();
	SDL_Surface *s = importer ->load_image(argc[1]);

	ImageManager::calculate_edge(s,MAPTOMIX_USE_SOBEL,MAPTOMIX_REPEAT);

	ImageManager::set_contrast(s,atoi(argc[2]));

	
	ImageManager::calculate_normal_map(s,atoi(argc[3]),4);
	ImageManager::compute_dudv(s,1.1);	
	window->display_image(s);
	
	ImageImporter::save_image(s,"sobel");
	handler->main_loop();		



delete window;

handler->close(); 
importer->close(); 
IMG_Quit();
SDL_Quit();
return EXIT_SUCCESS ; 
}
