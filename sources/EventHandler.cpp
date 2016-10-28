#include "../includes/EventHandler.h"


namespace maptomix{
	


	EventHandler::EventHandler(SDL_Event &event){
		m_event = event; 
		for(int i = 0 ; i < number_event ; i++)
			event_array[i]=false ; 

		app_quit = false;

	}

	EventHandler::~EventHandler(){

	}

/*************************************************************************************************************/
	void EventHandler::event_loop(){
		while(SDL_PollEvent(&m_event)){
			if(m_event.type == SDL_QUIT || m_event.key.keysym.sym == SDLK_ESCAPE)
			{
				event_array[ESC] = true ; 
				event_array[QUIT] = true;
				app_quit=true;
				
			}


		}


	}


/*************************************************************************************************************/

	void EventHandler::main_loop(){
		while(app_quit == false)
			event_loop();


		

	}




}
