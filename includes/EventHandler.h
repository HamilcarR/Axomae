#ifndef EVENTHANDLER_H
#define EVENTHANDLER_H
#include <SDL2/SDL.h>
#include <vector>
	


const bool ESC = 0 ; 	
const bool QUIT = 1 ;
const int number_event = 2; 


namespace maptomix{
	
	
	class EventHandler
	{

		public:
			EventHandler(SDL_Event &event);
			~EventHandler();
			void event_loop();
			void main_loop();
			bool* get_event_array(){return event_array;};

		private:
			SDL_Event m_event;
			bool event_array[number_event];
			bool app_quit ; 









	};










}













#endif
