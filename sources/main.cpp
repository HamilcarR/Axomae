#include <iostream>
#include <cstdlib>
#include <regex>
#include <string>
#include <thread> 
#include "../includes/ImageManager.h"
#include "../includes/ImageImporter.h"
#include "../includes/Window.h"
#include "../includes/EventHandler.h"
#include "../includes/TerminalOpt.h"

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


void quit_api(){
	

		IMG_Quit();
		SDL_Quit();

}


static void loop_event(EventHandler* ev){
	ev->main_loop(); 
}
int main(int argv , char** argc){
	init_api();

	ProgramStatus * main_program_command = ProgramStatus::getInstance();

	if (argv > 2) {
		string mode = argc[1];
		regex cmd, gui;
		try {
			cmd = regex("-cmd", regex_constants::icase);
		}
		catch (const std::regex_error& e) { cout << e.what() << "\n"; }
		if (regex_match(mode, cmd)) {
			bool ex = false;
			string user_input;

			while (!ex) {



#ifdef __unix__
				//	event_thread=thread(loop_event,event);
				cout << colors[GREEN] << prompt[0] << colors[YELLOW];
				std::getline(std::cin, user_input);
				ex = (regex_match(user_input, regex(command[EXIT], regex_constants::icase)));
				if (!ex)
					main_program_command->process_command(user_input);

				//	event_thread.join(); 	
#elif defined(_WIN32) || defined (WIN32)


				print(std::string(prompt[0]), YELLOW);
				std::getline(std::cin, user_input);
				ex = (regex_match(user_input, regex(command[EXIT], regex_constants::icase)));
				if (!ex)
					main_program_command->process_command(user_input);

				print(std::string(), RESET);



#endif

			}

		}
		else if (regex_match(mode, gui)) {



		}
		else {
			cout << "Wrong argument used" << "\n";
			return EXIT_FAILURE;
		}
	}
	else {
		cout << "Wrong argument used" << "\n";
	}


	
        

	main_program_command->exit(); 

	quit_api(); 
return EXIT_SUCCESS ; 
}
