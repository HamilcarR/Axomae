#ifndef EVENTHANDLER_H
#define EVENTHANDLER_H
#include <SDL2/SDL.h>
#include <vector>

#include "Controller.h"
	


 enum KEYS {ESC = 0 , QUIT = 1 };
typedef enum KEYS KEYS;
const int number_event = 2; 


namespace maptomix{
	
	
	class EventHandler : public maptomix::Controller
	{

		public:

			static EventHandler* getInstance();
			void setEvent(SDL_Event &event);
			void event_loop();
			void main_loop();
			bool* get_event_array(){return event_array;};
			void close(){delete instance;};

		private:
			static EventHandler* instance;
			~EventHandler();
			EventHandler();
			EventHandler(SDL_Event &event);


			
			SDL_Event m_event;
			bool event_array[number_event];
			bool app_quit ; 









	};










}






#endif
