#include "../includes/EventHandler.h"
#include <iostream>

namespace maptomix{
using namespace std;
EventHandler* EventHandler::instance = nullptr;

/********************************************************************************/
	EventHandler::EventHandler():Controller(){				//
		for(int i = 0 ; i < number_event ; i++){			//
			event_array[i]=false;					//
		}								//
										//
										//
		app_quit = false;						//
										//
	}									//
										//
/********************************************************************************/
										//
	EventHandler::EventHandler(SDL_Event &event):maptomix::Controller(){	//
		m_event = event; 						//
		for(int i = 0 ; i < number_event ; i++)				//
			event_array[i]=false ; 					//
										//
		app_quit = false;						//
										//
	}									//
/********************************************************************************/
	EventHandler::~EventHandler(){						//
										//
	}									//
										//
/*************************************************************************************************************/
	void EventHandler::event_loop(){
		while(SDL_PollEvent(&m_event)){
			if(m_event.type == SDL_QUIT )
			{

				event_array[QUIT] = true;
				cout<<"QUIT" <<endl; 
				app_quit=true;
				
			}
			else if( m_event.key.keysym.sym == SDLK_ESCAPE){

				
				cout<<"ESCP" <<endl; 
			//	app_quit=true;
				
				event_array[ESC] = true ; 
			}


		}


	}


/*************************************************************************************************************/

	void EventHandler::main_loop(){
		while(app_quit == false)
			event_loop();


		

	}



/*************************************************************************************************************/


	EventHandler* EventHandler::getInstance(){
		if(instance==nullptr)
			instance = new EventHandler();
		return instance;


	}














/*************************************************************************************************************/

	void EventHandler::setEvent(SDL_Event &event){
		m_event=event;


	}

	
	
	
/*************************************************************************************************************/
/*************************************************************************************************************/
}
