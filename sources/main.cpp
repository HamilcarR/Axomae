#include <iostream>
#include <cstdlib>
#include "../includes/Window.h"
#include "../includes/EventHandler.h"
using namespace std;
using namespace maptomix;
void init_api(){
	if(SDL_Init(SDL_INIT_EVERYTHING)!=0)
	{
		cout<<"SDL_Init problem"<<endl; 

	}

}





int main(int argv , char** argc){
	init_api();
	Window *window = new Window(400,400,"ok");
	SDL_Event event; 
	EventHandler *handler=new EventHandler(event);
	handler->main_loop();


delete window;
delete handler;
SDL_Quit();
return EXIT_SUCCESS ; 
}
