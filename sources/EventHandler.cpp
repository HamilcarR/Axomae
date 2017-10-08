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
	
	int EventHandler::get_pushed_button()
	{

		if( m_event.key.keysym.sym == SDLK_ESCAPE)
			return ESC;
		else if (m_event.key.keysym.sym == SDLK_UP)
			return UP;
		else if (m_event.key.keysym.sym == SDLK_DOWN)
			return DOWN;








	}


	void EventHandler::event_loop(){
		while(SDL_PollEvent(&m_event)){
			if(m_event.type == SDL_QUIT )
			{

				event_array[QUIT] = true;
				cout<<"QUIT" <<endl; 
				app_quit=true;
				
			}
			if(m_event.type == SDL_KEYDOWN){
				event_array[get_pushed_button()] = true;			
			}
		
		}


	}


/*************************************************************************************************************/

	void EventHandler::main_loop( ){
	
		while(app_quit == false){
			event_loop();
			reset_array();	
			SDL_Delay(100);
		}


		

	}


/***************************************************************************************************************/

	void EventHandler::reset_array(){
		
		fill(event_array,event_array+number_event,false);
		


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
