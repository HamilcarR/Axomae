#include <iostream>
#include <cstdlib>
#include <regex>
#include <string>
#include <thread> 

#include "../includes/ImageManager.h"
#include "../includes/ImageImporter.h"
#include "../includes/Window.h"
#include "../includes/TerminalOpt.h"
#include "../includes/GUIWindow.h"
using namespace std;
using namespace axomae;













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



int main(int argv , char** argc){
	init_api();

	ProgramStatus * main_program_command = ProgramStatus::getInstance();
	
	if (argv >= 2) {
		string mode = argc[1];
		regex cmd, gui;
		try {
			cmd = regex("-cmd", regex_constants::icase);
			gui = regex("-gui", regex_constants::icase); 
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
			main_program_command->exit();

		}
		else if (regex_match(mode, gui)) {
			
			QApplication app(argv, argc); 
			GUIWindow win; 
			win.show(); 
			

			return app.exec(); 

		}
		else {
			cout << "Wrong argument used" << "\n";
			return EXIT_FAILURE;
		}
	}
	else {
		cout << "Wrong command line argument used : Use -cmd for terminal or -gui for a graphical user interface" << "\n";
	}


	
        

	

	quit_api(); 
return EXIT_SUCCESS ; 
}
