#include <iostream>
#include <cstdlib>
#include <regex>
#include <signal.h>
#include <string>
#include <thread> 
#include <gtest/gtest.h>

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

void sigsegv_handler(int signal){	
	try{
		LOG("Application crash" , LogLevel::CRITICAL); 
		LOGFLUSH(); 
	}
	catch(const std::exception &e){
		std::cerr << e.what(); 
	}
	abort(); 

}

void quit_api(){
	IMG_Quit();
	SDL_Quit();
}


int main(int argv , char** argc){
	signal(SIGSEGV , sigsegv_handler);
	init_api();	
	ProgramStatus * main_program_command = ProgramStatus::getInstance();
	if (argv >= 2) {
		string mode = argc[1];
		regex cmd, test;
		try {
			cmd = regex("-cmd", regex_constants::icase);
			test = regex("-test", regex_constants::icase); 
		}
		catch (const std::regex_error& e) { cout << e.what() << "\n"; }
		if (regex_match(mode, cmd)) {
			bool ex = false;
			string user_input;
			while (!ex) {
#ifdef __unix__
				cout << colors[GREEN] << prompt[0] << colors[YELLOW];
				std::getline(std::cin, user_input);
				ex = (regex_match(user_input, regex(command[EXIT], regex_constants::icase)));
				if (!ex)
					main_program_command->process_command(user_input);
#elif defined(_WIN32) || defined (WIN32)

				print(std::string(prompt[0]), YELLOW);
				std::getline(std::cin, user_input);
				ex = (regex_match(user_input, regex(command[EXIT], regex_constants::icase)));
				if (!ex)
					main_program_command->process_command(user_input);
				print(std::string(), RESET)
#endif
			}
			main_program_command->exit();
		}
		else if (regex_match(mode, test)) {
			::testing::InitGoogleTest(&argv , argc);
			auto a = RUN_ALL_TESTS(); 
			return a; 
		}
		else {
			QApplication app(argv , argc);
			Controller win ; 
			std::string param_string = "" ; 
			for(int i = 1 ; i < argv ; i++)
				param_string += argc[i] + std::string(" ") ; 
			win.setApplicationConfig(param_string);
			win.show(); 
			return app.exec();
		}
	}
	else{
		QApplication app(argv, argc);		
		Controller win; 
		win.show(); 
		return app.exec(); 
	}
	quit_api(); 
	return EXIT_SUCCESS ; 
}
